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
    var percentage: Double
    
    init(isLoading: Bool = false,
         percentage: Double = 0.0) {
      self.isLoading = isLoading
      self.percentage = percentage
    }
  }
  
  var importData: Data?
  var loading: Loading
  var message: String
  var dropState: DropState
  var dashPhase: CGFloat
  var graphView: GraphView?
  
  init(importData: Data? = nil,
       loading: Loading = .init(),
       message: String = "",
       dropState: DropState = .none,
       dashPhase: CGFloat = 0.0,
       graphView: GraphView? = nil) {
    self.importData = importData
    self.loading = loading
    self.message = message
    self.dropState = dropState
    self.dashPhase = dashPhase
    self.graphView = graphView
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
          RoundedRectangle(cornerRadius: 12, style: .continuous)
            .strokeBorder(style: .init(lineWidth: 3, dash: [10.0], dashPhase: viewModel.dashPhase))
            .fill(.gray.opacity(0.3))
        }
        .padding()
        .opacity(viewModel.dropState == .enter ? 0.5 : 1.0)
        .animation(.easeInOut, value: viewModel.dropState == .enter)
      Color.gray.frame(height: 2)
      HStack {
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
        viewModel.graphView
      }
    }
    .onDrop(of: [.data], delegate: module)
  }
}
