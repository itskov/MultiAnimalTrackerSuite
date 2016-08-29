function [ ] = trackerCrawler( sourceDirectory, targetDirectory )
    % First we're looking for movie files (*.mj2)
    lister = FileLister(sourceDirectory,'*.mj2');
    videoFiles = lister.allFiles();
    numberOfMovies = length(videoFiles);
    
    % Then we're looking for Features files
    lister = FileLister(sourceDirectory,'*-Features.mat');
    featuresFiles = lister.allFiles();
    featuresFilesNames = {featuresFiles.name};
    
    % Sorting the movies by date.
    numDates = zeros(1,numberOfMovies);
    for i=1:numberOfMovies
        numDates(i) = datenum(videoFiles(i).date);
    end
    
    [~,ind] = sort(numDates,'descend');
    videoFiles = videoFiles(ind);
    
    % Going over the video file.
    parfor i=1:length(videoFiles)
        videoFile = videoFiles(i);
        [fileDir, fileName, ~] = fileparts(videoFile.name);
        
        % First we check if directory with such name already appears in the
        % target directory
        if (exist(fullfile(targetDirectory, fileName),'dir') == 7)
            continue;
        end
        
        % Look for the features file.
        currentFeaturesFileName = fullfile(fileDir, [fileName '-Features.mat']);
        currentFeatureFileInd = strcmp(featuresFilesNames, currentFeaturesFileName);
        
        if (~any(currentFeatureFileInd))
            continue
        end
        
        % Loading the features 
        currentFeatureFileName = featuresFilesNames(currentFeatureFileInd);
        currentFeatureFileName = currentFeatureFileName{1}; 
        [logStds,allFeatures] = loadMatFile(currentFeatureFileName);
        
        % Creating the directory
        newDir = fullfile(targetDirectory, fileName);
        mkdir(newDir);
        % Copying the features file.
        copyfile(currentFeatureFileName, newDir);
        
        % Starting the videoTracking
        %tracker = VideoTracker(videoFile.name,3000,fileName);
        tracker = VideoTracker(videoFile.name,Inf,fileName);
        
        tracker.performTracking3(allFeatures, logStds);
        tracks = tracker.tracks;
        
        saveMatFile(fullfile(newDir,['TrackingInfo.mat']),tracker, tracks);
    end

end

function [logStds,allFeatures] = loadMatFile(matFileName)
    load(matFileName);
end

function [] = saveMatFile(fileName, tracker, tracks)
    save(fileName,'tracker', 'tracks');
end
