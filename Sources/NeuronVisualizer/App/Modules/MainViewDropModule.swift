//
//  MainViewDropDelegate.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import SwiftUI

@available(macOS 14, *)
struct MainViewDropModule: DropDelegate {
  let viewModel: MainViewModel
  private let builder: Builder
  
  init(viewModel: MainViewModel,
       builder: Builder) {
    self.viewModel = viewModel
    self.builder = builder
  }
  
  func build(_ data: Data) async -> BuilderResult {
    await builder.build(data)
  }
  
  func performDrop(items: [NSItemProvider]) {
    guard let data = items.first else { return }
    
    let _ = data.loadDataRepresentation(for: .data) { data, error in
      self.viewModel.loading.isLoading = true
      self.viewModel.importData = data
    }
  }
  // MARK: DropDelegate
  
  func dropEntered(info: DropInfo) {
    // Triggered when an object enters the view.
    viewModel.dropState = .enter
  }
  
  func dropExited(info: DropInfo) {
    // Triggered when an object exits the view.
    viewModel.dropState = .none
  }
  
  func dropUpdated(info: DropInfo) -> DropProposal? {
    // Triggered when an object moves within the view.
    .none
  }
  
  func validateDrop(info: DropInfo) -> Bool {
    // Determines whether to accept or reject the drop.
    info.hasItemsConforming(to: [.data])
  }
  
  func performDrop(info: DropInfo) -> Bool {
    // Handles the drop when the user drops an object onto the view.
    performDrop(items: info.itemProviders(for: [.data]))
    return info.hasItemsConforming(to: [.data])
  }
}
