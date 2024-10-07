//
//  main.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/4/24.
//

import Neuron
import AppKit
import SwiftUI

if #available(macOS 14, *) {
  let app = NSApplication.shared
  let delegate = AppDelegate()
  app.delegate = delegate
  app.run()
}
