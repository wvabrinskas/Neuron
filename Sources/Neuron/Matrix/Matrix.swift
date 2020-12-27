//
//  Matrix.swift
//
//  Created by strictlyswift on 9/8/17.
//

import Foundation

struct Matrix<T> : CustomStringConvertible, ExpressibleByArrayLiteral {
    var rows: Int
    var cols: Int
    let sparse: Bool = false
    typealias MatrixType = T
    var values: [T] = []
    
    init(_ rows: Int, _ cols: Int, populating: (Int,Int) -> T  ) {
        self.rows = rows
        self.cols = cols
        
        values.reserveCapacity(rows*cols)
        
        for i in 0..<(rows*cols) {
            self.values.append( populating(i / cols, i % cols) )
        }
    }
    
    init(_ rows: Int, _ cols: Int, constant: T  ) {
        self.rows = rows
        self.cols = cols
        
        self.values = Array<T>(repeating: constant, count: rows*cols)
    }
    
    init(arrayLiteral: [T]...)  {
        self.rows = arrayLiteral.count
        self.cols = arrayLiteral[0].count
        
        values = arrayLiteral.flatMap { $0 }
    }
    
    init(withArray values: [T], rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        self.values = values
    }
    
    init(_ matrix: Matrix<T>) {
        self.rows = matrix.rows
        self.cols = matrix.cols
        self.values = matrix.values
    }
    
    init(_ array: [[T]])  {
        assert(array.count > 0, "Array must be of form [[]]")
        
        self.rows = array.count
        self.cols = array[0].count
        
        values = array.flatMap { $0 }
    }
    
    subscript( _ row: Int, _ col: Int ) -> T {
        get {
            assert( row < rows && col < cols, "Row and column index must be within bounds of matrix")
            
            return values[ (row * cols) + col ]
        }
        set {
            assert(row >= 0 && row < rows && col >= 0 && col < cols,"Row and column index must be within bounds of matrix")
            values[ (row * cols) + col ] = newValue
        }
    }
    
    var description : String {
        var s = "< "
        for row in 0..<rows {
            for col in 0..<cols {
                var t = ""
                let v = self[row, col]
                print( "\(v) ", terminator:"", to: &t)
                s += t
            }
            var t = ""
            if row != rows-1 { print("; ", terminator:"", to: &t) }
            
            s += t
        }
        return s + ">"
    }
    
    func map<S>( _ mapper: (Int, Int, T) -> S ) -> Matrix<S> {
        var newValues : [S] = []
        for (i,v) in values.enumerated() {
            let (row, col) = ( i/self.cols, i%self.cols )
            newValues.append(mapper(row, col, v))
        }
        return Matrix<S>(withArray:newValues, rows: self.rows, cols: self.cols)
    }
    
    func map<S,U>(on:Matrix<U>, _ mapper: (T, U) -> S ) -> Matrix<S> {
        assert( on.rows == self.rows && on.cols == self.cols, "Matrices are not the same size")
        
        var newValues : [S] = []
        newValues.reserveCapacity(values.count)
        for (i,v) in values.enumerated() {
            let (row, col) = ( i/self.cols, i%self.cols )
            newValues.append( mapper(v, on[row,col]) )
        }
        
        return Matrix<S>(withArray: newValues, rows: self.rows, cols: self.cols)
    }
    
    func mapEach<S>(_ mapper: (T) -> S ) -> Matrix<S> {
        return Matrix<S>(withArray: self.values.map (mapper), rows: self.rows, cols: self.cols)
    }
    
    func transpose() -> Matrix<T> {
        //   0 1 2      to    0 3
        //   3 4 5            1 4
        //                    2 5
        let first = self.values.first
        var newMatrix = Matrix<T>(self.cols, self.rows, constant: first! )
        for c in 0..<self.cols {
            for r in 0..<self.rows {
                newMatrix[c,r] = self[r,c]
            }
        }
        return newMatrix
    }
    
    func T() -> Matrix<T> {
        return self.transpose()
    }
    
    mutating func appendRow( _ rowVector: Matrix<T> ) {
        assert( rowVector.rows == 1, "appendRow must pass in row vector, ie with 1 row")
        assert( rowVector.cols == self.cols, "appendRow vector must be same width as original matrix")
        values.append(contentsOf: rowVector.values )
        self.rows += 1
    }
    
    mutating func appendCol( _ colVector: Matrix<T> ) {
        assert( colVector.cols == 1, "appendCol must pass in column vector, ie with 1 column")
        assert( colVector.rows == self.rows, "appendCol vector must be same height as original matrix")
        for i in 0..<self.rows {
            values.insert(colVector.values[i], at: self.cols * (i+1) )
        }
        self.cols += 1
    }
    
    func rowAt(_ row: Int) -> Matrix<T> {
        let r : Int
        if row < 0 {
            r = self.rows + row
        } else {
            r = row
        }
        let vals : [T] = Array(values[ r*self.cols ..< (r*(self.cols+1)) ])
        return Matrix<T>( [ vals ])
    }
    
    func colAt(_ col: Int) -> Matrix<T> {
        let c : Int
        if col < 0 {
            c = self.cols + col
        } else {
            c = col
        }
        let vals : [T] = values.enumerated().filter { n,x in n % self.cols == c }.map { $0.1 }
        return Matrix<T>( vals.map { [$0] } )
    }
}

extension Matrix where T : Comparable {
    func findMax() -> (max: T?, row: Int, col: Int) {
        var max: T? = nil
        var maxAt: Int = 0
        for (at,val) in values.enumerated() {
            if max == nil {
                max = val
                maxAt = at
            } else {
                if val > max! {
                    max = val
                    maxAt = at
                }
            }
        }
        return ( max: max, row: maxAt / self.cols, col: maxAt % self.cols )
    }
}

