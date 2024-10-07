//
//  MainView.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/4/24.
//

import SwiftUI
import AppKit
@_spi(Visualizer) import Neuron

@available(macOS 14, *)
@Observable
final class MainViewModel: Sendable {
  enum DropState {
    case enter, none
  }
  
  struct Loading {
    var isLoading: Bool
    var percentage: CGFloat
    
    init(isLoading: Bool = false,
         percentage: CGFloat = 0.0) {
      self.isLoading = isLoading
      self.percentage = percentage
    }
  }
  
  var importData: Data?
  var loading: Loading
  var message: String
  var dropState: DropState
  var dashPhase: CGFloat
  
  init(importData: Data? = nil,
       loading: Loading = .init(),
       message: String = "",
       dropState: DropState = .none,
       dashPhase: CGFloat = 0.0) {
    self.importData = importData
    self.loading = loading
    self.message = message
    self.dropState = dropState
    self.dashPhase = dashPhase
  }
}

@available(macOS 14, *)
struct MainView: View {
  @State private var viewModel: MainViewModel
  private var module: MainViewDropModule
  
  init(viewModel: MainViewModel, module: MainViewDropModule) {
    self.viewModel = viewModel
    self.module = module
  }
  
  var body: some View {
    VStack {
      Text("Drag Neuron model here")
        .font(.title)
        .padding(viewModel.dropState == .enter ? 20 : 16)
        .background {
          RoundedRectangle(cornerRadius: 20, style: .continuous)
            .strokeBorder(style: .init(lineWidth: 3, dash: [10.0], dashPhase: viewModel.dashPhase))
        }
        .padding()
        .opacity(viewModel.dropState == .enter ? 0.5 : 1.0)
        .animation(.easeInOut, value: viewModel.dropState == .enter)
      ScrollView {
        Text(viewModel.message)
          .padding([.leading, .trailing], 8)
        Spacer()
      }
      .overlay {
        if viewModel.loading.isLoading {
          ProgressView()
        }
      }
    }
    .onDrop(of: [.data], delegate: module)
    .task(id: viewModel.loading.isLoading) {
      guard let importData = viewModel.importData else { return }
      let buildResult = await module.build(importData)
      
      viewModel.message = buildResult.description
      viewModel.importData = nil
      viewModel.loading.isLoading = false
    }
  }
}

extension Animation {
  func `repeat`(while expression: Bool, autoreverses: Bool = true) -> Animation {
    if expression {
      return self.repeatForever(autoreverses: autoreverses)
    } else {
      return self
    }
  }
}

