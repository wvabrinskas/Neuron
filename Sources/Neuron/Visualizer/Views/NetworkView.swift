//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/18/22.
//

import Foundation
import SwiftUI


public struct NetworkView: View {
  public var viewModel: NetworkViewModel = NetworkViewModel()
  public init(viewModel: NetworkViewModel) {
    self.viewModel = viewModel
  }
  
  public var body: some View {
    BrainView(viewModel: viewModel.brain)
      .scalable()
  }
}
