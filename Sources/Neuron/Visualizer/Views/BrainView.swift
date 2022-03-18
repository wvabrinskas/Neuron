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
      VStack {
        ScrollView(.horizontal, showsIndicators: true) {
          HStack(spacing: viewModel.spacing) {
            ForEach(viewModel.lobes) { lobe in
              LobeView(viewModel: lobe)
            }
          }
          .padding()
        }
      }
    }
  }
}


struct BrainView_Previews: PreviewProvider {
  
  static private var lobes: [LobeViewModel] {
    let layerNum = Int.random(in: 3...5)
    
    var lobes: [LobeViewModel] = []
    var previousNum: Int = 0
    for _ in 0..<layerNum {
      
      let num = Int.random(in: 1...10)
      var models: [NeuronViewModel] = []
      for _ in 0..<num {
        
        var weights: [Float] = []
        for _ in 0..<previousNum {
          weights.append(Float.random(in: 0...1))
        }
        
        let model = NeuronViewModel(activation: Float.random(in: 0...1),
                                    weights: weights)
        models.append(model)
      }
      
      previousNum = num
      
      lobes.append(LobeViewModel(neurons: models, spacing: 15))
    }
    
    return lobes
  }
  
  
  static var previews: some View {
    if #available(iOS 15.0, *) {
      BrainView(viewModel: BrainViewModel(lobes: lobes, spacing: 100))
        .previewInterfaceOrientation(.landscapeLeft)
    } else {
      BrainView(viewModel: BrainViewModel(lobes: lobes))
      // Fallback on earlier versions
    }
  }
}

