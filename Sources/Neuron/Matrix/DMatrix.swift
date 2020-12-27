//
//  DMatrix.swift
//  MachineLearning
//
//  Created by strictlyswift on 9/10/17.
//

import Foundation
import Accelerate

typealias DMatrix = Matrix<Double>



indirect enum MatrixCalc {
    case m(DMatrix)
    case transpose(DMatrix)
    case dot(MatrixCalc, MatrixCalc)
    case sub(MatrixCalc, MatrixCalc)
    case add(MatrixCalc, MatrixCalc)
    case constMult(MatrixCalc, Double)
    
    func fast() -> DMatrix {
        switch self {
        case let .m(matrix): return matrix
            
        // the cases below have no special leaf-level processing, so use generic case matches
        case let .sub( a, b ) : return a.fast().fastSub( b.fast() )
        case let .constMult( a, value ) : return a.fast().fastMult(value)
            
        // where we have explicit leaf-level processing, use specific case matches
        // ...for "dot" multiplication
        case let .dot( .m(a), .m(b) ) : return a.dot(b)
        case let .dot( .m(a), .transpose( b) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: false, dot: b, transposeDot: true, beta: 0, add: nil)
        case let .dot( .transpose( a ), .m(b) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: true, dot: b, transposeDot: false, beta: 0, add: nil)
        case let .dot( .transpose( a ), .transpose(b) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: true, dot: b, transposeDot: true, beta: 0, add: nil)
            
       // ...for addition
        case let .add( .m(a), .m(b) ) : return a.fastAdd(b)
        case let .add( .dot( .m(a), .m(b) ), .m(c) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: false, dot: b, transposeDot: false, beta: 1.0, add: c)
        case let .add( .dot( .m(a), .transpose(b) ), .m(c) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: false, dot: b, transposeDot: true, beta: 1.0, add: c)
        case let .add( .dot( .transpose(a), .m(b)), .m(c) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: true, dot: b, transposeDot: false, beta: 1.0, add: c)
        case let .add( .dot( .transpose(a), .transpose(b) ), .m(c) ) : return a.generalDotAdd(alpha: 1.0, transposeSelf: true, dot: b, transposeDot: true, beta: 1.0, add: c)
            
        default: fatalError("Could not calculate matrix")
        }
    }
    
    static func +(a: MatrixCalc, b: DMatrix) -> MatrixCalc {
        return .add( a, .m(b) )
    }
    
    static func +(a: MatrixCalc, b: MatrixCalc) -> MatrixCalc {
        return .add( a, b )
    }
    
    static func ●(a: MatrixCalc, b: MatrixCalc) -> MatrixCalc {
        return .dot(a, b)
    }
    
    static func ●(a: MatrixCalc, b: DMatrix) -> MatrixCalc {
        return .dot( a, .m(b)) 
    }
    
    static func -(a: MatrixCalc, b: DMatrix) -> MatrixCalc {
        return .sub( a, .m(b))
    }
    
    static func -(a: DMatrix, b: MatrixCalc) -> MatrixCalc {
        return .sub( .m(a), b)
    }
    
    static func *(a: MatrixCalc, b: Double) -> MatrixCalc {
        return .constMult( a, b )
    }
    
    static func *(a: Double, b: MatrixCalc) -> MatrixCalc {
        return .constMult( b, a )
    }
}

extension Matrix where T == Double {

    func printFormatted() -> Void {
        var s = "< "
        for row in 0..<rows {
            for col in 0..<cols {
                let val = self[row,col]
                print( String(format:"%@%.2f ", (val<0 ? "" : " "),val), terminator: "" , to: &s)
            }
            print(s)
            s = "  "
        }
        print(">")
    }
    
    func dot(_ other: Matrix<Double>) -> Matrix<Double>
    {
        return generalDotAdd(alpha: 1.0, transposeSelf: false, dot: other, transposeDot: false, beta: 0.0, add: nil)
    }
    
    func dotAdd(dot: Matrix<Double>, add:Matrix<Double>) -> Matrix<Double> {
        return generalDotAdd(alpha: 1.0, transposeSelf: false, dot: dot, transposeDot: false, beta: 1.0, add: add)
    }
    
    /// Calculates  alpha(self ● dot) + beta(add).   Either self or dot may be transposed
    func generalDotAdd(alpha: Double, transposeSelf: Bool, dot: Matrix<Double>, transposeDot: Bool, beta: Double, add:Matrix<Double>?) -> Matrix<Double> {
        
        let selfAngle = (rows:(transposeSelf ? self.cols : self.rows), cols:(transposeSelf ? self.rows : self.cols))
        let dotAngle = (rows:(transposeDot ? dot.cols : dot.rows), cols:(transposeDot ? dot.rows: dot.cols ))
        assert( selfAngle.cols == dotAngle.rows, "Matrices are not compatible for multiplication")
        
        var m = Matrix<Double>( selfAngle.rows, dotAngle.cols, constant: 0.0)
        if add != nil {
            assert( selfAngle.rows == add!.rows && dotAngle.cols == add!.cols, "Matrices are not compatible for addition")
            m.values = add!.values
        } 
        
        cblas_dgemm(CblasRowMajor, transposeSelf ? CblasTrans : CblasNoTrans, transposeDot ? CblasTrans : CblasNoTrans, Int32(selfAngle.rows), Int32(dotAngle.cols), Int32(selfAngle.cols), alpha, self.values, Int32(self.cols), dot.values, Int32(dot.cols), beta, &m.values, Int32(m.cols))
        
        return m
    }
    
    func fastApply(other: DMatrix, with f: (Double, Double) -> Double, name: String) -> DMatrix {
        assert( self.rows == other.rows && self.cols == other.cols, "Matrices are not compatible for \(name)")
        var m = Matrix<Double>( self.rows, self.cols, constant: 0.0)
        for i in 0..<self.values.count {
            m.values[i] = f( self.values[i], other.values[i] )
        }
        
        return m
    }
    
    /// calculates self + b  quickly by avoiding the memory overhead of the zip
    func fastAdd(_ b: DMatrix) -> DMatrix { return self.fastApply(other: b, with: (+), name: "addition") }
    
    /// calculates self - b  quickly by avoiding the memory overhead of the zip
    func fastSub(_ b: DMatrix) -> DMatrix { return self.fastApply(other: b, with: (-), name: "subtraction") }

    /// calculates self ⊙ b  quickly by avoiding the memory overhead of the zip
    func fastHadamard(_ b: DMatrix) -> DMatrix { return self.fastApply(other: b, with: (*), name: "Hadamard") }

    mutating func inplaceAdd(_ other: DMatrix) {
        assert( self.rows == other.rows && self.cols == other.cols, "Matrices are not compatible for addition")
        for i in 0..<self.values.count {
            self.values[i] = self.values[i] + other.values[i]
        }
    }
    
    func transpose() -> MatrixCalc {
        return .transpose( self)
    }

    /// calculates self * value  quickly using BLAS
    func fastMult(_ b: Double) -> DMatrix {
        var m = Matrix<Double>( self.rows, self.cols, constant: 0.0)
        m.values = self.values

        cblas_dscal(Int32(self.rows * self.cols), b, &m.values, 1)
        return m
    }
    
    static func ●(a: DMatrix, b: DMatrix) -> MatrixCalc {
        return .dot( .m(a), .m(b) )
    }
    
    static func ●(a: DMatrix, b: MatrixCalc) -> MatrixCalc {
        return .dot( .m(a), b )
    }
    
    static func +(a: DMatrix, b: DMatrix) -> MatrixCalc {
        return .add( .m(a), .m(b) )
    }
    
    static func -(a: DMatrix, b: DMatrix) -> MatrixCalc {
        return .sub( .m(a), .m(b) )
    }
    
    static func *(a: DMatrix, b: Double) -> MatrixCalc {
        return .constMult( .m(a), b )
    }
    
    static func *(a: Double, b: DMatrix) -> MatrixCalc {
        return .constMult( .m(b), a )
    }
    
    static func ⊙(a: DMatrix, b: DMatrix) -> DMatrix {   // should really be MatrixCalc
        return a.fastHadamard(b)
    }
    
    static func +=(a: inout DMatrix, b: DMatrix) -> Void {   // should really be MatrixCalc
        return a.inplaceAdd(b)
    }
}
