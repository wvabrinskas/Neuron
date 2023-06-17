//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/16/23.
//

import Foundation

//@attached(conformance)
@attached(member)
public macro Layerable<T: RawRepresentable>(type: T,
                                            inputSize: [Float] = [0,0,0],
                                            outputSize: [Float] = [0,0,0]) = #externalMacro(module: "MacrosImpl", type: "LayerMacro")
