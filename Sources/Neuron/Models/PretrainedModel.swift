//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/21/21.
//

import Foundation

public struct PretrainedModel: ModelBuilder {
  public var fileURL: String
  
  public init(url file: String) {
    self.fileURL = file
  }
  
  internal func getModel() -> Result<ExportModel?, Error> {
    let modelJsonResult: Result<[AnyHashable: Any]?, Error> = self.getJSON(fileURL)
    
    do {
      let modelJSON = try modelJsonResult.get()
      let modelResult: Result<ExportModel?, Error> = self.build(modelJSON)
      let model = try modelResult.get()
      
      return .success(model)
      
    } catch {
      return .failure(error)
    }
  }
}

