//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/18/22.
//

import Foundation
import SwiftUI

public struct ScalableView: ViewModifier {
  @State private var scale: CGFloat = 1.0
  @State private var lastScale: CGFloat = 1.0
  @State private var viewState = CGSize.zero
  
  @State var magScale: CGFloat = 1
  @State var progressingScale: CGFloat = 1
  
  var magnification: some Gesture {
    MagnificationGesture()
      .onChanged { progressingScale = $0 }
      .onEnded {
        magScale *= $0
        progressingScale = 1
      }
  }
  
  public func body(content: Content) -> some View {
    content
      .animation(.spring(), value: 0)
      .offset(x: viewState.width, y: viewState.height)
      .gesture(DragGesture()
        .onChanged { val in
          self.viewState = val.translation
        }
      )
      .scaleEffect(self.magScale * progressingScale)
      .gesture(magnification)
  }
}

public extension View {
  func scalable() -> some View {
    modifier(ScalableView())
  }
}
