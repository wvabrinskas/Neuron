//
//  MainViewDropDelegate.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import SwiftUI

@available(macOS 14, *)
final class MainViewDropModule: DropDelegate {
  let viewModel: MainViewModel
  private let builder: Builder
  
  init(viewModel: MainViewModel,
       builder: Builder) {
    self.viewModel = viewModel
    self.builder = builder
  }
  
  func build(_ data: Data?) async {
    guard let data else { return }
    
    let buildResult = await builder.build(data)
    
    viewModel.message = buildResult.description
    clean()
  }
  
  func performDrop(items: [NSItemProvider]) {
    viewModel.message.removeAll()
    
    guard let data = items.first else { return }
    
    let _ = data.loadDataRepresentation(for: .data) { data, error in
      self.viewModel.loading.isLoading = true
      Task { @MainActor in
        await self.build(data)
      }
    }
  }
  
  private func clean() {
    viewModel.importData = nil
    viewModel.loading = .init()
  }
  
  // MARK: DropDelegate
  
  func onBuildComplete() {
    
  }
  
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
    info.hasItemsConforming(to: [.data]) && viewModel.loading.isLoading == false
  }
  
  func performDrop(info: DropInfo) -> Bool {
    guard viewModel.loading.isLoading == false else { return false }
    // Handles the drop when the user drops an object onto the view.
    performDrop(items: info.itemProviders(for: [.data]))
    return info.hasItemsConforming(to: [.data])
  }
}
