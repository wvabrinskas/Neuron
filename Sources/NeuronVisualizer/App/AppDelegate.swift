//
//  AppDelegate.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/4/24.
//

import Neuron
import AppKit
import SwiftUI

@available(macOS 14, *)
class WindowDelegate: NSObject, NSWindowDelegate {
  
  func windowWillClose(_ notification: Notification) {
    NSApplication.shared.terminate(0)
  }
}

@available(macOS 14, *)
class AppDelegate: NSObject, NSApplicationDelegate {
  let window = NSWindow()
  let windowDelegate = WindowDelegate()
  
  let mainViewModel: MainViewModel = .init()
  
  func applicationDidFinishLaunching(_ notification: Notification) {
    let appMenu = NSMenuItem()
    appMenu.submenu = NSMenu()
    appMenu.submenu?.addItem(NSMenuItem(title: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
    let mainMenu = NSMenu(title: "Neuron Network Visualizer")
    mainMenu.addItem(appMenu)
    NSApplication.shared.mainMenu = mainMenu
    
    let size = CGSize(width: 480, height: 270)
    window.setContentSize(size)
    window.styleMask = [.closable, .miniaturizable, .resizable, .titled]
    window.delegate = windowDelegate
    window.title = "Neuron Network Visualizer"
      
    let module = MainViewDropModule(viewModel: mainViewModel,
                                    builder: Builder())
    
    let view = NSHostingView(rootView: MainView(viewModel: module.viewModel,
                                                module: module))
    
    view.frame = CGRect(origin: .zero, size: size)
    view.autoresizingMask = [.height, .width]
    window.contentView!.addSubview(view)
    window.center()
    window.makeKeyAndOrderFront(window)
    
    NSApp.setActivationPolicy(.regular)
    NSApp.activate(ignoringOtherApps: true)
  }
}
