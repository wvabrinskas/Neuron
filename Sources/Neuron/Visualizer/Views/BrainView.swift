//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/17/22.
//

import Foundation
import SwiftUI


public struct BrainView: View {
  public var viewModel: BrainViewModel
  
  public init(viewModel: BrainViewModel) {
    self.viewModel = viewModel
  }
   
  public var body: some View {
    ScrollView(.vertical, showsIndicators: true) {
      ScrollView(.horizontal, showsIndicators: true) {
        HStack(spacing: viewModel.spacing) {
          ForEach(viewModel.lobes) { lobe in
            Spacer()
            LobeView(viewModel: lobe)
            Spacer()
          }
        }
        .padding()
      }
    }
  }
}


struct BrainView_Previews: PreviewProvider {
  
  static private var lobes: [LobeViewModel] {
    let layerNum = Int.random(in: 3...6)
    
    var lobes: [LobeViewModel] = []
    for _ in 0..<layerNum {
      
      let num = Int.random(in: 1...10)
      var models: [NeuronViewModel] = []
      for _ in 0..<num {
        let model = NeuronViewModel(activation: Float.random(in: 0...1))
        models.append(model)
      }
      
      lobes.append(LobeViewModel(neurons: models, spacing: 15))
    }
    
    return lobes
  }
  
  
  static var previews: some View {
    if #available(iOS 15.0, *) {
      BrainView(viewModel: BrainViewModel(lobes: lobes))
        .previewInterfaceOrientation(.landscapeLeft)
    } else {
      BrainView(viewModel: BrainViewModel(lobes: lobes))
      // Fallback on earlier versions
    }
  }
}

