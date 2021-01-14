//
//  Algebra.swift
//
//  Created by strictlyswift on 9/8/17.
//

import Foundation


protocol Semiring {
    static func + (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    static var zero: Self { get }
    static var one: Self { get }
}

protocol Ring : Semiring {
    static func - (lhs: Self, rhs: Self) -> Self
    
}

extension Semiring {
    static func += (lhs: inout Self, rhs: Self) {
        lhs = lhs + rhs
    }
}

extension Double: Ring {
    static var zero: Double {
        return 0
    }
    
    static var one: Double {
        return 1
    }
}

extension Int : Ring {
    static var zero: Int {
        return 0
    }
    
    static var one: Int {
        return 1
    }
}

extension Bool : Semiring {
    static func + (lhs: Bool, rhs: Bool) -> Bool {
        return lhs || rhs
    }
    
    static func * (lhs: Bool, rhs: Bool) -> Bool {
        return lhs && rhs
    }
    
    static let zero = false
    static let one = true
    
}
