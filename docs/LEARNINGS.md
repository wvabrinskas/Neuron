# LEARNINGS

Deliberate trade-offs, deferred decisions, and constraints that shaped the current design.

This file is for things that are **correct but not ideal** — where a constraint forced a
compromise, and a future change to that constraint should prompt a revisit. It is not a bug
list (fix those) and not a changelog (that's `CLAUDE.md`'s Recent Changes section).

Each entry states what forced the decision, what the code does now, what it cost, and the
trigger that should make someone reconsider. When you take a shortcut you can defend but
don't love, add an entry rather than a `TODO` that nobody will find.

---

## `ignoreLabelIndex` lives on the Optimizer, not the LossFunction

**Context.** Sparse losses need to exclude padded timesteps from loss, gradients, and
accuracy. Conceptually that index belongs to the loss function — it is a property of how the
loss is computed, not of the optimizer.

Two things blocked putting it there:

1. `LossFunction` is an enum used as a plain value (`.sparseCrossEntropySoftmax`) across the
   codebase and in ~11 existing tests. Swift has no default values for associated values, so
   `case sparseCrossEntropySoftmax(ignoring: Int?)` breaks every one of those call sites.
2. More fundamentally, **the timing doesn't work**. `RNN` builds its `ClassifierParameters`
   (which carry `lossFunction`) in `init`, but the pad token ID isn't known until
   `readyUp()` has trained the tokenizer. A loss function that carried the index would have
   to be replaced after dataset construction, which means it can't be `let` on `Classifier`
   and the "loss is a value" model breaks anyway.

**Decision.** `Optimizer.ignoreLabelIndex: Int?` (settable, defaults to `nil`), threaded into
`LossFunction.calculate(_:correct:ignoring:)`, `.derivative(_:correct:ignoring:)`, and
`MetricCalculator.calculateAccuracy`. `RNN.readyUp()` sets it from `dataset.padTokenId` once
the tokenizer has trained.

**Cost.** The ignore index is configured on a different object than the loss it modifies, and
it is invisible to anyone reading the loss function alone. `Optimizer` gains one more piece
of training-policy state it doesn't conceptually own.

**Revisit when.** Training configuration is refactored so loss and data are constructed
together — a `TrainingSession`-style object that owns dataset, loss, and optimizer would let
the ignore index sit with the loss where it belongs. Alternatively, if `LossFunction` ever
becomes a protocol with concrete types instead of an enum, the index becomes a stored
property on the sparse implementations and the timing problem is solved by constructing the
loss after the dataset.

---

## `RNN.predict` works around lossy decoding instead of fixing `decode`

**Context.** `BPETokenizer.decode` replaces `</w>` with a space and then calls
`trimmingCharacters(in: .whitespaces)`. Decoding one token at a time and concatenating the
results therefore loses every word boundary — `"the cat sat"` comes back as `"thecatsat"`.

**Decision.** `RNN.predict` accumulates token **IDs** during generation and decodes the whole
sequence in a single pass at the end (`assemble(tokenIds:delimiter:)`). `decode` itself is
unchanged.

**Cost.** The invariant "decode is only correct on a complete sequence" is implicit. Any new
caller that decodes token-by-token and joins will silently hit the same bug. It is covered by
a test (`TokenizableDatasetTests.test_item_decodesWholeSequenceWithWordBoundaries`) but not by
the type system.

**Revisit when.** A second caller needs incremental decoding — streaming generation, for
instance. The fix is a decoder that carries state across calls (emitting the pending space
when the *next* token arrives) rather than trimming each fragment independently.

---

## `RNN.compile` validates the dataset instead of padding it

**Context.** `LSTM.forward` iterates a fixed `0..<batchLength`, zero-filling timesteps a short
sample doesn't provide and never reading the ones a long sample carries past the window.
Neither is reported, so a ragged dataset trains on silently corrupted sequences. BPE makes
datasets ragged by default — equal character counts no longer imply equal token counts.

The RNN *could* pad the data itself: `dataset.padTokenId` is reachable.

**Decision.** `RNN.compile` calls `validate(dataset:wordLength:)` and fails with a message
naming the offending sample. It does not modify the data.

**Cost.** Callers have to pad correctly themselves, and a mistake is a hard failure rather
than something the framework absorbs. `TokenizableDataset.nextTokenPair(for:sequenceLength:)`
exists to make the correct path easy, but nothing forces its use.

**Revisit when.** This should probably stay. Silently reshaping a caller's training data is
how you end up debugging a model that trained on something other than what you handed it. If
it changes, it should be an explicit opt-in (`RNN(padSequences: true)`), never the default.

---

## `BPETokenizer` builds its base vocabulary through `Vectorizer`

**Context.** `BPETokenizer.train` seeds its vocabulary by calling `vectorizer.vectorize` for
the special tokens and then the corpus characters, and copies `vectorizer.vector` into
`vocab`. Merge tokens are then added to `vocab` only — the `Vectorizer` never sees them.

This split ownership caused two shipped bugs: `nextId` was seeded from
`vectorizer.lastKey + 1` (`lastKey` is already the next free slot, so an ID was skipped), and
`vocabSize` returned `vectorizer.vector.count`, which doesn't count merge tokens at all.
Together they let merge IDs index past the end of the embedding table.

**Decision.** `nextId` is derived (`vocab.count`) rather than stored, and `vocabSize` is
`vocab.count`, so "IDs are contiguous in `0..<vocabSize`" holds by construction. The
`Vectorizer` dependency remains for seeding.

**Cost.** Two objects still hold overlapping vocabulary state, and only one of them is the
source of truth. `Vectorizer.lastKey`/`maxIndex` semantics (post-increment, reset per call)
are easy to misread and were the origin of the off-by-one.

**Revisit when.** `BPETokenizer` needs anything else from the vocabulary that `Vectorizer`
doesn't provide. Assigning base IDs directly is a handful of lines and removes the
dependency entirely; `Vectorizer` is a character-level tool that BPE has outgrown.

---

## `BPETokenizer.vocab` / `reverseVocab` are internal rather than private

**Context.** The invariant that matters is "every ID in `0..<vocabSize` is assigned". A decode
round-trip cannot verify it: IDs like `</w>` and `" "` decode to whitespace that gets trimmed,
so an unassigned ID and a whitespace-rendering ID are indistinguishable from outside.

**Decision.** Both dictionaries are `private(set)` internal, and
`TokenizerTests.test_train_assignsContiguousIds` asserts
`Set(reverseVocab.keys) == Set(0..<vocabSize)` directly.

**Cost.** Two implementation details are visible to the rest of the module. They are
read-only, so the risk is coupling rather than corruption.

**Revisit when.** A public vocabulary-inspection API is wanted for its own sake (listing
merges, dumping the vocabulary). `token(for id: Int) -> String?` plus a count would cover both
the test and the use case, and these could go back to `private`.
