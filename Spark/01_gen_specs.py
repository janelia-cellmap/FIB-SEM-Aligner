#!/usr/bin/env python

import glob, json, os, re

baseDataPath = "/groups/hess/hesslab/Cryo_data/EM_data/20161213/20161213_S2_Cell13_Raw_Tif_500MHz_60um_D10-D19"
tileSpecFilePath = "sections.json"

# 0: tileId, 1: z, 2: imagePath  
tileSpecTemplate = """{{
  "tileId" : "{0}.{1}.0",
  "layout" : {{
    "sectionId" : "{1}.0",
    "temca" : "0",
    "camera" : "0",
    "imageRow" : 0,
    "imageCol" : 0,
    "stageX" : 0.0,
    "stageY" : 0.0,
    "rotation" : 0.0
  }},
  "z" : {1}.0,
  "width" : 18750.0,
  "height" : 1500.0,
  "minIntensity" : 0.0,
  "maxIntensity" : 255.0,
  "mipmapLevels" : {{
    "0" : {{
      "imageUrl" : "file:{2}"
    }}
  }},
  "transforms" : {{
    "type" : "list",
    "specList" : [ {{
      "type" : "leaf",
      "className" : "mpicbg.trakem2.transform.AffineModel2D",
      "dataString" : "1 0 0 1 0 0"
    }} ]
  }}
}}"""

tileSpecList = []

# NOTE: first z must be greater than 0 for Matlab processes to work
z = 1
for imagePath in sorted(glob.glob(baseDataPath + "/*.tif")):
    # NVision40-3802_<date>_<time>_0-0-0_InLens.tif
    baseName = os.path.basename(imagePath)
    baseNamePieces = baseName.split('_')
    tileId = "%s_%s" % (baseNamePieces[1], baseNamePieces[2])
    tileSpecList.append(tileSpecTemplate.format(tileId, z, imagePath))
    z = z + 1
    
print "Found %d .tif files in %s" % (len(tileSpecList), baseDataPath)

if len(tileSpecList) > 0:
    tileSpecFile = open(tileSpecFilePath, 'w')
    tileSpecFile.write('[\n')
    count = 0
    for tileSpec in tileSpecList:
        if count > 0:
            tileSpecFile.write(',\n')
        tileSpecFile.write(tileSpec)
        count = count + 1
    tileSpecFile.write('\n]')       

    print "Wrote %d tile specs to %s" % (len(tileSpecList), tileSpecFilePath)
