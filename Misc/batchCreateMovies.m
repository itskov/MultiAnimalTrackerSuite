function [] = batchCreateMovies( sourceDirectory )
%BATCHATTRACTIONFIELD Summary of this function goes here
%   Detailed explanation goes here
lister = FileLister(sourceDirectory,'TrackingInfo.mat');
trackingInfoFiles = lister.allFiles();


for i=1:length(trackingInfoFiles)
    % Closing all opened figures;
    close all;
    
    trackingInfoFile = trackingInfoFiles(i);
    [fileDir, ~, ~] = fileparts(trackingInfoFile.name);
    

    disp(trackingInfoFile.name);
    
    % Loading the tracking Info.
    load(trackingInfoFile.name);
    

     
    fileName = fullfile(fileDir, [tracker.name '.avi']);
    if (exist(fileName,'file'))
        continue;
    end
    
    tracker.viewTracks(tracker.tracks,[1 tracker.numberOfFrames],0,fileDir);
    
end




end
