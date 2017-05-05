% Converts *.dat raw FIB-SEM data files to 16bit or 8bit tiff images,
% followed by alignment using SIFT
%
% Rev history 10/13/2014
%   1st rev.
% 10/22/2014
%   added file pattern filter
% 7/4/2015
%   added support for file version 8 16-bit data
% 11/15/2016
%   revised matlabpool to parpool for v2016b, avoid cd into the directory
%   contains all *.dat files to prevent Matlab overhead
% 11/16/2016
%   corrected conversion of standard transformation matrix by SIFT to the one
%   of imod with includes frame center coordinates
% 11/17/2016
%   added combo transform: translation only but substitute with affine when
%   rotation more than 0.5 degrees
% 1/30/2017
%   added -Xmx4g -XX:+UseCompressedOops for fiji (need to remove -Xmx flag
%   in /usr/local/Fiji.app/ImageJ.cfg)


clearvars;

Dat=0; RawStack=0;

%% Read in 3DSEM *.dat file
[FileNames,PathName] = uigetfile(...
  {'*.dat; *.tif; *.png; *.jpg','Data Files or Image files (*.dat, *.tif, *.png, *.jpg)';...
  '*.dat','Select data files (*.dat)';...
  '*.tif', 'Select tif files (*.tif)';...
  '*.png', 'Select png files (*.png)';...
  '*.jpg', 'Select jpg files (*.jpg)';...
  '*.*','All Files (*.*)'},...
  'Multiselect','on',...
  'Select Data Files'); % Display standard dialog box to select files
if not(strcmp(PathName(length(PathName)),'/')), PathName=[PathName, '/']; end
if ischar(FileNames) && ~isempty(regexpi(FileNames,'FileList')) % extract file names from the file name list
  temp=load([PathName FileNames]);
  FileNames=temp.FileNames;
  clear temp;
end
if length(FileNames)>1
    FileNames=sort(FileNames);
    FileNamePairs=vertcat(FileNames(1:end-1),FileNames(2:end));
end

%% Remove files match certain patterns
% FileNames=FileNames(cellfun(@isempty,regexpi(FileNames(:),'_0-0-0.')));

%% Save images

FileNumber=size(FileNames,2); % Number of files

if length(char(FileNames(1)))-regexpi(char(FileNames(1)),'.dat')==3
  Dat=1;
  FIBSEMData=readfibsem([PathName char(FileNames(1))]);
  Atiffsuff=['_' deblank(FIBSEMData.DetA) '.tif'];
  Btiffsuff=['_' deblank(FIBSEMData.DetB) '.tif'];
  FirstImg=regexprep(FileNames{1},'.dat',Atiffsuff);
  parfor FileN=1:FileNumber % convert *.dat to *.tif
    AN=0; BN=0; ARaw=0; BRaw=0; % reset flags for available images
    
    if FileNumber==1
      FileName=char(FileNames);
    else
      FileName=char(FileNames(FileN));
    end
    FIBSEMData=readfibsem([PathName FileName]); % script to read *.dat files
    
    % generate normalized images ImageAN and ImageBN, as well as raw 16 bit
    % images ImageARaw and ImageBRaw if available.
    if FIBSEMData.EightBit==1 % 8-bit *.dat
      if FIBSEMData.AI1
        AN=1;
        ImageAN=FIBSEMData.ImageA;
      end
      if FIBSEMData.AI2
        BN=1;
        ImageBN=FIBSEMData.ImageB;
      end
    else % 16-bit *.dat
      switch FIBSEMData.FileVersion
        case {1,2,3,4,5,6}
          if FIBSEMData.AI1
            AN=1;
            ImageAN=uint16((FIBSEMData.ImageA+10)/20*65535); % normalize image data to uint16
          end
          if FIBSEMData.AI2
            BN=1;
            ImageBN=uint16((FIBSEMData.ImageB+10)/20*65535); % normalize image data to uint16
          end
        otherwise
          if FIBSEMData.AI1
            AN=1;
            ImageAN=uint16(FIBSEMData.ImageA);
            ARaw=1;
            ImageARaw=uint16(single(FIBSEMData.RawImageA)+32768); % convert raw int16 data to uint16
          end
          if FIBSEMData.AI2
            BN=1;
            ImageBN=uint16(FIBSEMData.ImageB);
            BRaw=1;
            ImageBRaw=uint16(single(FIBSEMData.RawImageB)+32768);
          end
      end
    end
    
    % save images with electron counts (scaling factor in Image Description)
    if AN==1
      imwrite(ImageAN,[PathName regexprep(FileName,'.dat',Atiffsuff)],...
        'Description',['Electron scaling factor: ', num2str(1/FIBSEMData.Scaling(4,1))],...
        'Resolution',1/FIBSEMData.PixelSize*2.54*10^7);
    end
    if BN==1
      imwrite(ImageBN,[PathName regexprep(FileName,'.dat',Btiffsuff)],...
        'Description',['Electron scaling factor: ', num2str(1/FIBSEMData.Scaling(4,2))],...
        'Resolution',1/FIBSEMData.PixelSize*2.54*10^7);
    end
  end
end

if length(char(FileNames(1)))-regexpi(char(FileNames(1)),'.png')==3
  FirstImg=regexprep(FileNames{1},'.png','.tif');
  parfor FileN=1:FileNumber
    FileName=char(FileNames(FileN));
    Image=imread([PathName FileName]);
    imwrite(Image,[PathName regexprep(FileName,'.png','.tif')]);
    
    fprintf(1,'%s%g%s%g%s\n','File ',FileN,' of ',FileNumber,' converted to tif.');
    
  end
end

if length(char(FileNames(1)))-regexpi(char(FileNames(1)),'.tif')==3
  FirstImg=FileNames{1};
end

%% Copy Fiji SIFT align script to working directory
switch computer
  case {'PCWIN','PCWIN64'}
  
  case {'MACI','MACI64'}
    !cp ~/Documents/MATLAB/"Fiji SIFT align.bsh" .
  case {'GLNX86','GLNXA64'}
    !cp "/usr/local/bin/Fiji SIFT align.bsh" .
  otherwise
end

%% Calculate transformation matrix of each image relative to the previous one
Img=imread([PathName FirstImg]);
[ImgY,ImgX]=size(Img);
ImgZ=FileNumber;

SIFTalignAffxf=zeros(FileNumber,6);
SIFTalignAffxf(1,:)=[1 0 0 1 0 0];
SIFTalignTransxf=SIFTalignAffxf;

parfor_progress(FileNumber-1);
parfor FileN=2:FileNumber
  [PreFileName,FileName]=FileNamePairs{:,FileN-1};
  if Dat==1
    FileName=regexprep(FileName,'.dat',Atiffsuff);
    PreFileName=regexprep(PreFileName,'.dat',Atiffsuff);
  end
  system(['ImageJ -Xmx4g -XX:+UseCompressedOops -Dpre="' [PathName PreFileName] '" -Dpost="' [PathName FileName] '" -- --headless "Fiji SIFT align.bsh" > /dev/null 2>&1']);
  SIFTFileName=[PathName FileName '-SIFT.txt'];
  fid=fopen(SIFTFileName);
  A=textscan(fid,'%s','delimiter','\n','whitespace','');
  fclose(fid);
  Affine=char(A{1}{1});
  Affine=strsplit(Affine,{'[[',']]'});
  Affine=regexprep(Affine{2},{'[',']',','},'');
  Affine=textscan(Affine,'%f');
  Affine=Affine{1};
  AffDX=Affine(3)-(1-Affine(1))*ImgX/2+Affine(2)*ImgY/2;
  AffDY=-Affine(6)+(1-Affine(5))*ImgY/2-Affine(4)*ImgX/2;
  
  SIFTalignAffxf(FileN,:)=[Affine(1) -Affine(2) -Affine(4) Affine(5) AffDX AffDY];
  Trans=char(A{1}{2});
  Trans=strsplit(Trans,{'[[',']]'});
  Trans=regexprep(Trans{2},{'[',']',','},'');
  Trans=textscan(Trans,'%f');
  Trans=Trans{1};
  TransDX=Trans(3)-(1-Trans(1))*ImgX/2+Trans(2)*ImgY/2;
  TransDY=-Trans(6)+(1-Trans(5))*ImgY/2-Trans(4)*ImgX/2;
  SIFTalignTransxf(FileN,:)=[Trans(1) -Trans(2) -Trans(4) Trans(5) TransDX TransDY];
  system(['rm "' SIFTFileName '"']);
  parfor_progress;  % Count
end
parfor_progress(0); % Clean up

% replaced Translation only coefficients with Affine coefficients for
% frames rotate more than 0.5 degrees.
SIFTalignComboxf=SIFTalignTransxf;
SIFTalignComboxf(abs(SIFTalignAffxf(:,3))>0.0008,:)=SIFTalignAffxf(abs(SIFTalignAffxf(:,3))>0.0008,:);

%% Generate xf files
fid=fopen([PathName, 'SIFTalignAff.xf'],'w');
fprintf(fid,'%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n', SIFTalignAffxf');
fclose(fid);
fid=fopen([PathName, 'SIFTalignTrans.xf'],'w');
fprintf(fid,'%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n', SIFTalignTransxf');
fclose(fid);
fid=fopen([PathName, 'SIFTalignCombo.xf'],'w');
fprintf(fid,'%12.7f%12.7f%12.7f%12.7f%12.3f%12.3f\n', SIFTalignComboxf');
fclose(fid);

!rm "Fiji SIFT align.bsh"
%% Convert tif to mrc and generate aligned mrc file using xg transformation
if exist([PathName 'rawstack.mrc'],'file')==2 % evaluates if rawstack.mrc exists
  fid=fopen([PathName, 'rawstack.mrc'],'r');
  fseek(fid,0,'bof');
  nx=fread(fid,1,'int32');
  ny=fread(fid,1,'int32');
  nz=fread(fid,1,'int32');
  type=fread(fid,1,'int32');
  fclose(fid);
  if nx==ImgX && ny==ImgY && nz==ImgZ, RawStack=1; end  % rawstack.mrc exists
end
if RawStack==0  % generates rawstack.mrc if needed
  system(['tif2mrc -g "' PathName '"*InLens.tif "' PathName 'rawstack.mrc"']);
end
system(['xftoxg -nfit 0 "' PathName '"SIFTalignCombo.xf "' PathName '"SIFTalignCombo.xg']);
system(['newstack -xform "' PathName '"SIFTalignCombo.xg -mode 0 -scale 0,255 "' PathName '"rawstack.mrc "' PathName '"SIFTalignCombo.mrc']);
system(['xftoxg -nfit 0 "' PathName '"SIFTalignTrans.xf "' PathName '"SIFTalignTrans.xg']);
system(['newstack -xform "' PathName '"SIFTalignTrans.xg -mode 0 -scale 0,255 "' PathName '"rawstack.mrc "' PathName '"SIFTalignTrans.mrc']);
% system(['xftoxg -nfit 0 "' PathName '"SIFTalignAff.xf "' PathName '"SIFTalignAff.xg']);
% system(['newstack -xform "' PathName '"SIFTalignAff.xg -mode 0 -scale 0,255 "' PathName '"rawstack.mrc "' PathName '"SIFTalignAff.mrc']);
