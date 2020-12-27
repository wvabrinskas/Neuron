//
//  Matrix_Algebra.swift
//
//  Created by strictlyswift on 9/8/17.
//

import Foundation


infix operator ● : MultiplicationPrecedence
infix operator ⊙ : MultiplicationPrecedence

extension Matrix where T : Semiring {
    func dot(_ other: Matrix<MatrixType>) -> Matrix<MatrixType>
    {
        
        assert( self.cols == other.rows, "Matrices are not compatible for multiplication")
        var m : Matrix<T> = Matrix<T>(self.rows, other.cols, constant: MatrixType.zero)
        
        // have to use MatrixType above (a typealias for T) as otherwise the complier gets confused about exactly which T
        
        for sr in 0..<self.rows {
            for oc in 0..<other.cols {
                for i in 0..<self.cols {
                    m[sr, oc] += ( self[sr, i] * other[i, oc] )
                }
            }
        }
        
        return m
    }
    
    func add(_ other: Matrix<T>) -> Matrix<T> {
        return self.map(on: other) { $0 + $1 }
    }
    
    static func ●(a: Matrix<T>, b: Matrix<T>) -> Matrix<T> {
        return a.dot(b)
    }
    
    static func +(a: Matrix<T>, b: Matrix<T>) -> Matrix<T> {
        return a.add(b)
    }
    
    static func I(_ size:Int) -> Matrix<T> {
        return Matrix<T>(size, size) { row, col in ( row == col) ? MatrixType.one : MatrixType.zero }
    }
    
    static func ⊙(a: Matrix<T>, b: Matrix<T>) -> Matrix<T> {
        return a.map(on: b) { $0 * $1 }
    }
    
    static func *(a: Matrix<T>, b: T) -> Matrix<T> {
        return a.mapEach { $0 * b }
    }
    
    static func *(a: T, b: Matrix<T>) -> Matrix<T> {
        return b.mapEach { $0 * a }
    }
    
    static func zero(_ rows: Int, _ cols: Int)  -> Matrix<T>  {
        return Matrix<T>(rows, cols, constant:  MatrixType.zero)
    }
}

extension Matrix where T : Ring {
    static func -(a: Matrix<T>, b: T) -> Matrix<T> {
        return a.mapEach { $0 - b }
    }
    
    static func -(a: Matrix<T>, b: Matrix<T>) -> Matrix<T> {
        return a.map(on: b) { $0 - $1 }
    }
    
}


