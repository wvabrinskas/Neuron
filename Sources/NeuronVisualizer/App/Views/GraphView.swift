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
    ScrollView([.horizontal, .vertical]) {
      HStack(alignment: .center) {
        let buildView = buildViews(node: root)
        ForEach(0..<buildView.count, id: \.self) { i in
          let view = buildView[i]
          view
        }
      }
      .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
    }
  
  }
    
  func buildViews(node: Node, buildSelf: Bool = true, runningViews: [AnyView] = []) -> [AnyView] {
    guard node.connections.isEmpty == false else {
      return runningViews
    }
    
    var views: [AnyView] = runningViews
    
    // build self
    if buildSelf {
      views.append(AnyView(node.build()))
    }
    
    let viewToAppend = AnyView(VStack {
      ForEach(0..<node.connections.count, id: \.self) { i in
        let nodeToConnect = node.connections[i]
        AnyView(nodeToConnect.build())
      }
    })
    
    views.append(viewToAppend)
    
    // build children
    for childNode in node.connections {
      views = buildViews(node: childNode,
                         buildSelf: false,
                         runningViews: views)
    }
    
    return views
  }
}
