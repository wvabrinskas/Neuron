//
//  GraphView.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import SwiftUI


public struct GraphView: View {
  let root: Node
  
  public var body: some View {
    ScrollView(.vertical) {
      ScrollView(.horizontal) {
        HStack(alignment: .center) {
          let buildView = buildViews()
          ForEach(0..<buildView.count, id: \.self) { i in
            let view = buildView[i]
            view
          }
        }
        .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
      }
    }
  }
  
  // TODO: show output shape maybe?
  
  func buildViews() -> [AnyView] {
    var views: [AnyView] = []
    
    var node: Node? = root
    
    while node != nil {
      let build = node?.build()
      
      if let build {
        views.append(AnyView(build))
      }
      
      node = node?.connections.first
    }
    
    return views
  }
}
