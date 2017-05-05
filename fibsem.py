#!/usr/bin/env python
# -*- coding: utf-8 -*-
# fibsem.py
"""
Functions for reading FIB-SEM data from Hess Lab's proprietary format

Copyright (c) 2017, David Hoffman
"""
import os
import numpy as np

class FIBSEMHeader(object):
    """Structure to hold header info"""
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def update(**kwargs):
        """update internal dictionary"""
        self.__dict__.update(kwargs)


class _DTypeDict(object):
    """Handle dtype dict manipulations"""
    def __init__(self, names=None, formats=None, offsets=None):
        # initialize internals to empty lists
        self.names = []
        self.formats = []
        self.offsets = []
        if names is not None:
            self.update(names, formats, offsets)

    def update(self, names, formats, offsets):
        """"""
        if isinstance(names, list):
            if len(names) == len(formats) == len(offsets):
                self.names.extend(names)
                self.formats.extend(formats)
                self.offsets.extend(offsets)
            else:
                raise RuntimeError("Lengths are not equal")
        else:
            self.names.append(names)
            self.formats.append(formats)
            self.offsets.append(offsets)

    @property
    def dict(self):
        """Return the dict representation"""
        return dict(names=self.names, formats=self.formats, offsets=self.offsets)

    @property
    def dtype(self):
        """return the dtype"""
        return np.dtype(self.dict)

def _read_header(fobj):
    # make emtpy header to fill 
    base_header_dtype = _DTypeDict()
    base_header_dtype.update(
            [
                "FileMagicNum", # Read in magic number, should be 3555587570
                "FileVersion", # Read in file version number
                "FileType", # Read in file type, 1 is Zeiss Neon detectors
                "SWdate", # Read in SW date
                "TimeStep", # Read in AI sampling time (including oversampling) in seconds
                "ChanNum", # Read in number of channels
                "EightBit" # Read in 8-bit data switch
            ],
            [">u4", ">u2", ">u2", ">S10", ">f8", ">u1", ">u1"],
            [0, 4, 6, 8, 24, 32, 33]
        )
    # read initial header
    base_header = np.fromfile(fobj, dtype=base_header_dtype.dtype, count=1)
    FIBSEMData = FIBSEMHeader(**dict(zip(base_header.dtype.names, base_header[0])))
    # now fobj is at position 34, return to 0
    fobj.seek(0, os.SEEK_SET)
    if FIBSEMData.FileMagicNum != 3555587570:
        raise RuntimeError("FileMagicNum should be 3555587570 but is {}".format(FIBSEMData.FileMagicNum))

    more_header_dict = {"names": [], "formats": [], "offsets": []}
    if FIBSEMData.FileVersion == 1:
        base_header_dtype.update("Scaling", ('>f8', (4, FIBSEMData.ChanNum)), 36)
    elif FIBSEMData.FileVersion in {2,3,4,5,6}:
        base_header_dtype.update("Scaling", ('>f4', (4, FIBSEMData.ChanNum)), 36)
    else:
        # Read in AI channel scaling factors, (col#: AI#), (row#: offset, gain, 2nd order, 3rd order)
        base_header_dtype.update("Scaling", ('>f4', (4, 2)), 36)

    base_header_dtype.update(
        ["XResolution",  # X Resolution
        "YResolution"],  # Y Resolution
        ['>u4', '>u4'], 
        [100, 104]
    )

    if FIBSEMData.FileVersion in {1,2,3}:
        base_header_dtype.update(
                [
                    "Oversampling",  # AI oversampling
                    "AIDelay",  # Read AI delay (
                ],
                [">u1", ">i2"],
                [108, 109]
            )
    else:
        base_header_dtype.update("Oversampling", '>u2', 108)  # AI oversampling

    base_header_dtype.update("ZeissScanSpeed", '>u1', 111) # Scan speed (Zeiss #)
    if FIBSEMData.FileVersion in {1,2,3}:
        base_header_dtype.update(
                [
                    "ScanRate",  # Actual AO (scanning) rate
                    "FramelineRampdownRatio",  # Frameline rampdown ratio
                    "Xmin",  # X coil minimum voltage
                    "Xmax",  # X coil maximum voltage
                ],
                [">f8", ">f8", ">f8", ">f8"],
                [112, 120, 128, 136]
            )
        # FIBSEMData.Detmin = -10 # Detector minimum voltage
        # FIBSEMData.Detmax = 10 # Detector maximum voltage
    else:
        base_header_dtype.update(
            [
                "ScanRate", # Actual AO (scanning) rate
                "FramelineRampdownRatio", # Frameline rampdown ratio
                "Xmin", # X coil minimum voltage
                "Xmax", # X coil maximum voltage
                "Detmin", # Detector minimum voltage
                "Detmax", # Detector maximum voltage
                "DecimatingFactor" # Decimating factor
            ],
            [">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">u2"],
            [112, 116, 120, 124, 128, 132, 136]
        )

    base_header_dtype.update(
        [
            "AI1", # AI Ch1
            "AI2", # AI Ch2
            "AI3", # AI Ch3
            "AI4", # AI Ch4
            "Notes", # Read in notes
        ],
        [">u1", ">u1", ">u1", ">u1", ">S200"],
        [151, 152, 153, 154, 180]
    )

    if FIBSEMData.FileVersion in {1, 2}:
        base_header_dtype.update(
                [
                    "DetA",  # Name of detector A
                    "DetB",  # Name of detector B
                    "DetC",  # Name of detector C
                    "DetD",  # Name of detector D
                    "Mag",  # Magnification
                    "PixelSize",  # Pixel size in nm
                    "WD",  # Working distance in mm
                    "EHT",  # EHT in kV
                    "SEMApr",  # SEM aperture number
                    "HighCurrent",  # high current mode (1=on, 0=off)
                    "SEMCurr",  # SEM probe current in A
                    "SEMRot",  # SEM scan roation in degree
                    "ChamVac",  # Chamber vacuum
                    "GunVac",  # E-gun vacuum
                    "SEMStiX",  # SEM stigmation X
                    "SEMStiY",  # SEM stigmation Y
                    "SEMAlnX",  # SEM aperture alignment X
                    "SEMAlnY",  # SEM aperture alignment Y
                    "StageX",  # Stage position X in mm
                    "StageY",  # Stage position Y in mm
                    "StageZ",  # Stage position Z in mm
                    "StageT",  # Stage position T in degree
                    "StageR",  # Stage position R in degree
                    "StageM",  # Stage position M in mm
                    "BrightnessA",  # Detector A brightness (
                    "ContrastA",  # Detector A contrast (
                    "BrightnessB",  # Detector B brightness (
                    "ContrastB",  # Detector B contrast (
                    "Mode",  # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External, 9=External+SEM
                    "FIBFocus",  # FIB focus in kV
                    "FIBProb",  # FIB probe number
                    "FIBCurr",  # FIB emission current
                    "FIBRot",  # FIB scan rotation
                    "FIBAlnX",  # FIB aperture alignment X
                    "FIBAlnY",  # FIB aperture alignment Y
                    "FIBStiX",  # FIB stigmation X
                    "FIBStiY",  # FIB stigmation Y
                    "FIBShiftX",  # FIB beam shift X in micron
                    "FIBShiftY",  # FIB beam shift Y in micron
                ],
                [
                    ">S10", ">S18", ">S20", ">S20", ">f8", ">f8", ">f8", ">f8", ">u1",
                    ">u1", ">f8", ">f8", ">f8", ">f8", ">f8", ">f8", ">f8",
                    ">f8", ">f8", ">f8", ">f8", ">f8", ">f8", ">f8", ">f8",
                    ">f8", ">f8", ">f8", ">u1", ">f8", ">u1", ">f8", ">f8",
                    ">f8", ">f8", ">f8", ">f8", ">f8", ">f8"
                ],
                [
                    380, 390, 700, 720, 408, 416, 424, 432, 440, 441, 448, 456,
                    464, 472, 480, 488, 496, 504, 512, 520, 528, 536, 544, 552,
                    560, 568, 576, 584, 600, 608, 616, 624, 632, 640, 648, 656,
                    664, 672, 680
                ]
            )
    else:
        base_header_dtype.update(
                [
                    "DetA",  # Name of detector A
                    "DetB",  # Name of detector B
                    "DetC",  # Name of detector C
                    "DetD",  # Name of detector D
                    "Mag",  # Magnification
                    "PixelSize",  # Pixel size in nm
                    "WD",  # Working distance in mm
                    "EHT",  # EHT in kV
                    "SEMApr",  # SEM aperture number
                    "HighCurrent",  # high current mode (1=on, 0=off)
                    "SEMCurr",  # SEM probe current in A
                    "SEMRot",  # SEM scan roation in degree
                    "ChamVac",  # Chamber vacuum
                    "GunVac",  # E-gun vacuum
                    "SEMShiftX",  # SEM beam shift X
                    "SEMShiftY",  # SEM beam shift Y
                    "SEMStiX",  # SEM stigmation X
                    "SEMStiY",  # SEM stigmation Y
                    "SEMAlnX",  # SEM aperture alignment X
                    "SEMAlnY",  # SEM aperture alignment Y
                    "StageX",  # Stage position X in mm
                    "StageY",  # Stage position Y in mm
                    "StageZ",  # Stage position Z in mm
                    "StageT",  # Stage position T in degree
                    "StageR",  # Stage position R in degree
                    "StageM",  # Stage position M in mm
                    "BrightnessA",  # Detector A brightness (#)
                    "ContrastA",  # Detector A contrast (#)
                    "BrightnessB",  # Detector B brightness (#)
                    "ContrastB",  # Detector B contrast (#)
                    "Mode",  # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External, 9=External+SEM
                    "FIBFocus",  # FIB focus in kV
                    "FIBProb",  # FIB probe number
                    "FIBCurr",  # FIB emission current
                    "FIBRot",  # FIB scan rotation
                    "FIBAlnX",  # FIB aperture alignment X
                    "FIBAlnY",  # FIB aperture alignment Y
                    "FIBStiX",  # FIB stigmation X
                    "FIBStiY",  # FIB stigmation Y
                    "FIBShiftX",  # FIB beam shift X in micron
                    "FIBShiftY",  # FIB beam shift Y in micron
                ],
                [
                    ">S10", ">S18", ">S20", ">S20", ">f4", ">f4", ">f4", ">f4", ">u1",
                    ">u1", ">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">f4",
                    ">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">f4",
                    ">f4", ">f4", ">f4", ">f4", ">f4", ">u1", ">f4", ">u1",
                    ">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">f4"
                ],
                [
                    380, 390, 410, 430, 460, 464, 468, 472, 480, 481, 490,
                    494, 498, 502, 510, 514, 518, 522, 526, 530, 534, 538,
                    542, 546, 550, 554, 560, 564, 568, 572, 600, 604, 608,
                    620, 624, 628, 632, 636, 640, 644, 648
                ]
            )

    if FIBSEMData.FileVersion in {5,6,7,8}:
        base_header_dtype.update(
                [
                    "MillingXResolution",  # FIB milling X resolution
                    "MillingYResolution",  # FIB milling Y resolution
                    "MillingXSize",  # FIB milling X size (um)
                    "MillingYSize",  # FIB milling Y size (um)
                    "MillingULAng",  # FIB milling upper left inner angle (deg)
                    "MillingURAng",  # FIB milling upper right inner angle (deg)
                    "MillingLineTime",  # FIB line milling time (s)
                    "FIBFOV",  # FIB FOV (um)
                    "MillingLinesPerImage",  # FIB milling lines per image
                    "MillingPIDOn",  # FIB milling PID on
                    "MillingPIDMeasured",  # FIB milling PID measured (0:specimen, 1:beamdump)
                    "MillingPIDTarget",  # FIB milling PID target
                    "MillingPIDTargetSlope",  # FIB milling PID target slope
                    "MillingPIDP",  # FIB milling PID P
                    "MillingPIDI",  # FIB milling PID I
                    "MillingPIDD",  # FIB milling PID D
                    "MachineID",  # Machine ID
                    "SEMSpecimenI",  # SEM specimen current (nA)
                ],
                [
                    ">u4", ">u4", ">f4", ">f4", ">f4", ">f4", ">f4",
                    ">f4", ">u2", ">u1", ">u1", ">f4", ">f4", ">f4",
                    ">f4", ">f4", ">S30", ">f4"
                ],
                [
                    652, 656, 660, 664, 668, 672, 676, 680, 684, 686,
                    689, 690, 694, 698, 702, 706, 800, 980
                ]
            )

    if FIBSEMData.FileVersion in {6,7}:
        base_header_dtype.update(
                [
                    "Temperature",  # Temperature (F)
                    "FaradayCupI",  # Faraday cup current (nA)
                    "FIBSpecimenI",  # FIB specimen current (nA)
                    "BeamDump1I",  # Beam dump 1 current (nA)
                    "SEMSpecimenI",  # SEM specimen current (nA)
                    "MillingYVoltage",  # Milling Y voltage (V)
                    "FocusIndex",  # Focus index
                    "FIBSliceNum",  # FIB slice #
                ],
                [">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">f4", ">u4"],
                [850, 854, 858, 862, 866, 870, 874, 878]
            )
    if FIBSEMData.FileVersion == 8:
        base_header_dtype.update(
                [
                    "BeamDump2I",  # Beam dump 2 current (nA)
                    "MillingI",  # Milling current (nA)
                ],
                [">f4", ">f4"],
                [882, 886]
            )
    base_header_dtype.update("FileLength", ">i8", 1000) # Read in file length in bytes
    
    # read header
    header = np.fromfile(fobj, dtype=base_header_dtype.dtype, count=1)
    FIBSEMData = FIBSEMHeader(**dict(zip(header.dtype.names, header[0])))

    return FIBSEMData


def _convert_data(Raw, FIBSEMData):
    """"""
    ## Convert raw data to electron counts
    if FIBSEMData.EightBit == 1:
        Scaled = np.empty(Raw.shape, dtype=np.float)
        DetectorA, DetectorB = Raw
        if FIBSEMData.AI1:
            DetectorA = Raw[0]
            Scaled[0] = (Raw[0] * FIBSEMData.ScanRate / FIBSEMData.Scaling[0, 0] / FIBSEMData.Scaling[2, 0] / FIBSEMData.Scaling[3, 0] + FIBSEMData.Scaling[1, 0])
            if FIBSEMData.AI2:
                DetectorB = Raw[1]
                Scaled[1] = Raw[1] * FIBSEMData.ScanRate / FIBSEMData.Scaling[0, 1] / FIBSEMData.Scaling[2, 1] / FIBSEMData.Scaling[3, 1] + FIBSEMData.Scaling[1, 1]
        
        elif FIBSEMData.AI2:
            DetectorB = Raw[0]
            Scaled[0] = (Raw[0] * FIBSEMData.ScanRate / FIBSEMData.Scaling[0, 0] / FIBSEMData.Scaling[2, 0] / FIBSEMData.Scaling[3, 0] + FIBSEMData.Scaling[1, 0])
        
    else:
        raise NotImplementedError("Don't support non-8 bit files")
    #     if FIBSEMData.FileVersion in {1,2,3,4,5,6}:
    #         if FIBSEMData.AI1:
    #             # Converts raw I16 data to voltage based on scaling factors
    #             DetectorA = FIBSEMData.Scaling(1,1)+single(Raw(:,1))*FIBSEMData.Scaling(2,1)
    #             if FIBSEMData.AI2:
    #                 # Converts raw I16 data to voltage based on scaling factors
    #                 DetectorB = FIBSEMData.Scaling(1,2)+single(Raw(:,2))*FIBSEMData.Scaling(2,2)
    #                 if FIBSEMData.AI3:
    #                     DetectorC = FIBSEMData.Scaling(1,3)+single(Raw(:,3))*FIBSEMData.Scaling(2,3)
    #                     if FIBSEMData.AI4:
    #                         DetectorD = FIBSEMData.Scaling(1,4)+single(Raw(:,4))*FIBSEMData.Scaling(2,4)
    #                 elif FIBSEMData.AI4:
    #                     DetectorD = FIBSEMData.Scaling(1,3)+single(Raw(:,3))*FIBSEMData.Scaling(2,3)
                    
    #             elif FIBSEMData.AI3:
    #                 DetectorC = FIBSEMData.Scaling(1,2)+single(Raw(:,2))*FIBSEMData.Scaling(2,2)
    #                 if FIBSEMData.AI4:
    #                     DetectorD = FIBSEMData.Scaling(1,3)+single(Raw(:,3))*FIBSEMData.Scaling(2,3)
                    
    #             elif FIBSEMData.AI4:
    #                 DetectorD = FIBSEMData.Scaling(1,2)+single(Raw(:,2))*FIBSEMData.Scaling(2,2)
                
    #         elif FIBSEMData.AI2:
    #             DetectorB = FIBSEMData.Scaling(1,1)+single(Raw(:,1))*FIBSEMData.Scaling(2,1)
    #             if FIBSEMData.AI3:
    #                 DetectorC = FIBSEMData.Scaling(1,2)+single(Raw(:,2))*FIBSEMData.Scaling(2,2)
    #                 if FIBSEMData.AI4:
    #                     DetectorD = FIBSEMData.Scaling(1,3)+single(Raw(:,3))*FIBSEMData.Scaling(2,3)
                    
    #             elif FIBSEMData.AI4:
    #                 DetectorD = FIBSEMData.Scaling(1,2)+single(Raw(:,2))*FIBSEMData.Scaling(2,2)
                
    #         elif FIBSEMData.AI3:
    #             DetectorC = FIBSEMData.Scaling(1,1)+single(Raw(:,1))*FIBSEMData.Scaling(2,1)
    #             if FIBSEMData.AI4:
    #                 DetectorD = FIBSEMData.Scaling(1,2)+single(Raw(:,2))*FIBSEMData.Scaling(2,2)
                
    #         elif FIBSEMData.AI4:
    #             DetectorD = FIBSEMData.Scaling(1,1)+single(Raw(:,1))*FIBSEMData.Scaling(2,1)
                
    #     if FIBSEMData.FileVersion in {7,8}:
    #         if FIBSEMData.AI1:
    #             # Converts raw I16 data to voltage based on scaling factors
    #             DetectorA = (single(Raw(:,1))-FIBSEMData.Scaling(2,1))*FIBSEMData.Scaling(3,1)
    #             if FIBSEMData.AI2:
    #                 DetectorB = (single(Raw(:,2))-FIBSEMData.Scaling(2,2))*FIBSEMData.Scaling(3,2)
                
    #         elif FIBSEMData.AI2:
    #             DetectorB = (single(Raw(:,1))-FIBSEMData.Scaling(2,2))*FIBSEMData.Scaling(3,2)

    return DetectorA, DetectorB, Scaled


def readfibsem(path):
    """Read raw data file (*.dat) generated from Neon
    Needs PathName and FileName

    Rev history
    04/17/09 
            1st rev.
    07/31/2011
            converted from script to function
    11/25/2012
            added support for file version 5
    6/20/2013
            read raw data up to
            [FIBSEMData.ChanNum,FIBSEMData.XResolution*FIBSEMData.YResolution]
    6/25/2013
            added support for file version 6
    7/10/2013
            added decimating factor
    7/1/2014
            added file version 7 for 8-bit data support
    7/4/2014
            added file version 8 support
    """

    ## Load raw data file 's' or 'ieee-be.l64' Big-ian ordering, 64-bit long data type
    with open(path, 'rb') as fobj: # Open the file written by LabView (big-ian byte ordering and 64-bit long data type)
        # read header
        FIBSEMData = _read_header(fobj)
    # read data
    if FIBSEMData.EightBit == 1:
        Raw = np.memmap(path, dtype=">u1", mode="r", offset=1024,
            shape=(FIBSEMData.YResolution, FIBSEMData.XResolution, FIBSEMData.ChanNum))
    else:
        Raw = np.memmap(path, dtype=">u2", mode="r", offset=1024,
            shape=(FIBSEMData.YResolution, FIBSEMData.XResolution, FIBSEMData.ChanNum))
    Raw = np.rollaxis(Raw, 2)

    return Raw
    
    # DetectorA, DetectorB, Scaled = _convert_data(Raw, FIBSEMData)

    # return DetectorA, DetectorB, Scaled[0], Scaled[1]

    ## Construct image files
    # if FIBSEMData.AI1:
    #     FIBSEMData.ImageA = (reshape(DetectorA,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #     FIBSEMData.RawImageA = (reshape(Raw(:,1),FIBSEMData.XResolution,FIBSEMData.YResolution))
    #     if FIBSEMData.AI2:
    #         FIBSEMData.ImageB = (reshape(DetectorB,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         FIBSEMData.RawImageB = (reshape(Raw(:,2),FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         if FIBSEMData.AI3:
    #             raise NotImplementedError
    #         #     FIBSEMData.ImageC = (reshape(DetectorC,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         #     FIBSEMData.RawImageC = (reshape(Raw(:,3),FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         #     if FIBSEMData.AI4:
    #         #         FIBSEMData.ImageD = (reshape(DetectorD,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         #         FIBSEMData.RawImageD = (reshape(Raw(:,4),FIBSEMData.XResolution,FIBSEMData.YResolution))
                
    #         elif FIBSEMData.AI4:
    #             raise NotImplementedError
    #             # FIBSEMData.ImageD = (reshape(DetectorD,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #             # FIBSEMData.RawImageD = (reshape(Raw(:,3),FIBSEMData.XResolution,FIBSEMData.YResolution))
            
    #     elif FIBSEMData.AI3:
    #         # FIBSEMData.ImageC = (reshape(DetectorC,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         # FIBSEMData.RawImageC = (reshape(Raw(:,2),FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         # if FIBSEMData.AI4:
    #         #     FIBSEMData.ImageD = (reshape(DetectorD,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         #     FIBSEMData.RawImageD = (reshape(Raw(:,3),FIBSEMData.XResolution,FIBSEMData.YResolution))
            
    #     elif FIBSEMData.AI4:
    #         # FIBSEMData.ImageD = (reshape(DetectorD,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         # FIBSEMData.RawImageD = (reshape(Raw(:,2),FIBSEMData.XResolution,FIBSEMData.YResolution))
        
    # elif FIBSEMData.AI2:
    #     FIBSEMData.ImageB = (reshape(DetectorB,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #     FIBSEMData.RawImageB = (reshape(Raw(:,1),FIBSEMData.XResolution,FIBSEMData.YResolution))
    #     if FIBSEMData.AI3:
    #         # FIBSEMData.ImageC = (reshape(DetectorC,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         # FIBSEMData.RawImageC = (reshape(Raw(:,2),FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         if FIBSEMData.AI4:
    #         #     FIBSEMData.ImageD = (reshape(DetectorD,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         #     FIBSEMData.RawImageD = (reshape(Raw(:,3),FIBSEMData.XResolution,FIBSEMData.YResolution))
            
    #     elif FIBSEMData.AI4:
    #         # FIBSEMData.ImageD = (reshape(DetectorD,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         # FIBSEMData.RawImageD = (reshape(Raw(:,2),FIBSEMData.XResolution,FIBSEMData.YResolution))
        
    # elif FIBSEMData.AI3:
    #     # FIBSEMData.ImageC = (reshape(DetectorC,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #     # FIBSEMData.RawImageC = (reshape(Raw(:,1),FIBSEMData.XResolution,FIBSEMData.YResolution))
    #     if FIBSEMData.AI4:
    #         # FIBSEMData.ImageD = (reshape(DetectorD,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #         # FIBSEMData.RawImageD = (reshape(Raw(:,2),FIBSEMData.XResolution,FIBSEMData.YResolution))
        
    # elif FIBSEMData.AI4:
    #     # FIBSEMData.ImageD = (reshape(DetectorD,FIBSEMData.XResolution,FIBSEMData.YResolution))
    #     # FIBSEMData.RawImageD = (reshape(Raw(:,1),FIBSEMData.XResolution,FIBSEMData.YResolution))
