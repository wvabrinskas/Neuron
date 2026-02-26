//
//  LayerGroup.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/25/26.
//

import NumSwift


protocol LayerGroup: AnyObject {
  func build() -> [Layer]
}
