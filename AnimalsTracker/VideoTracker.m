classdef VideoTracker < handle
    %VIDEOTRACKER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % The input file of the video
        inputFile
        
        % Input file name
        videoFileName
        
        % The Raw Data.
        videoRawData;
        
        % Flat field init mat
        flatFieldMat
        
        % Saving the last frames
        prevFrames
        
        % Video after tracking.
        trackedVideo
        
        % Number Of Frames
        numberOfFrames
        
        %All found Centroids
        allCentroids
        
        %All Found entites
        allEntities
        
        % Learner
        learner
        
        % Tracks
        tracks
        
        % Chemo attractant positions
        chemoCords
        
        % Base position.
        baseCords
        
        % Misc parameters
        parameters
        
        % Name.
        name
        
        % How many steps should we allow to predict
        ALLOW_PRED_STEPS = 10
        
        % Learning steps period
        LEARNING_STEPS = 10
        
        % on/off prediction
        SHOULD_PREDICT = true
        
        % Should perform ML object recognition.
        SHOULD_ML = true;
        
        % Empty track scaffhold.
        EMPTY_TRACK;
    end
    
    methods
        function VT = VideoTracker(fileName, numberOfFrames, name)
            if (length(fileName) == 0)
                return
            end
            
            if (~exist('numberOfFrames','var'))
                numberOfFrames = inf;
            end
            
            if (~exist('name','var'))
                [~,name,~] = fileparts(fileName);
            end
            
            % Initializing
            VT.chemoCords = [0 0];
            
            % Saving the name of the video.
            VT.name = name;
            
            % Saving the file name.
            VT.videoFileName = fileName;
            
            % Intializing.
            VT.initialize(numberOfFrames);            
        end
        
        function initialize(obj, numberOfFrames)
            HIGH_FRAME_RATE = 1;
            
            % Initialize the video object using a file name from disk.
            obj.inputFile = VideoReader(obj.videoFileName);
            
            if (nargin < 2)
                
                if (HIGH_FRAME_RATE)
                    obj.numberOfFrames = ...
                        floor(obj.inputFile.FrameRate * obj.inputFile.Duration);
                else
                % The number of frames in the video
                obj.numberOfFrames = obj.inputFile.NumberOfFrames;
                end
            else
                % If we have to calculate the number of frames by ourselves
                if isempty(obj.inputFile.NumberOfFrames)
                    fileNumberOfFrames = obj.inputFile.Duration * obj.inputFile.FrameRate;
                else
                    fileNumberOfFrames = obj.inputFile.NumberOfFrames;
                end
                
                % Saving the number of frames the user chooses.
                obj.numberOfFrames = floor(min(fileNumberOfFrames, numberOfFrames));
            end
            
            if (~HIGH_FRAME_RATE)
                obj.flatFieldMat = obj.initializeFlatField();
            end
        end
        
        % Setting the position of the chemoattractant points.
        % After setting it, we save it in all available tracks.
        function setChemoCords(obj)
            imshow(obj.getRawFrame(obj.numberOfFrames - 1));
            title(obj.name);
            hold on;
            
            [x,y,b] = ginput(3);
            
            if (b == 3)
                obj.chemoCords = [];
                return;
            else
                points = [x y];
                obj.chemoCords(1,:) = points(1,:);
                obj.chemoCords(2,:) =  points(2,:);
                obj.chemoCords(3,:) =  points(3,:);
            end
            
            % distance
            distance = pdist2(points(2,:),points(3,:));
            disp(distance);
            
            % Saving it to all of the tracks.
            for i=1:length(obj.tracks)
                obj.tracks(i).chemoCords = obj.chemoCords;
                obj.tracks(i).pixelsNormalization = distance;
            end
        end
        
        
        
        % Constructs an object with a file.
        function newObj = clone(obj)
            newObj = VideoTracker('');
            newObj.videoRawData = obj.videoRawData;
            newObj.numberOfFrames = obj.numberOfFrames;
        end
        
        
        
        function frame = getRawFrame(obj, frameIndex)
            % Returns a single frame from the video.
            %frame = obj.videoRawData(:,:,:,frameIndex);
            
            %frame = wiener2(read(obj.inputFile, frameIndex), [5 5]);
            frame = read(obj.inputFile, frameIndex);
            
            %DEBUG
            frame = uint8(frame);
            %DEBUG
                        
            % If we have 3 RGB channels.
            if (size(frame,3) == 3)
                % We convert to gray scale.
                frame = rgb2gray(frame);
            end
        end
        
        function frame = getFilteredFrame(obj, frameIndex, binaryThreshold)
            
            if (nargin < 3)
                binaryThreshold = 0.955;
            end
            
            
            frame = obj.getRawFrame(frameIndex);
            
            % Running filter
            frame = obj.applyFlatField(frame, obj.flatFieldMat);
            frame = obj.applyContrast(frame,binaryThreshold);
            
        end
        
        %         function [gaussianStd] = modifyGaussianStd2(obj, frameIndex, initialGaussianStd, targetThreshold)
        %             currentGaussianStd = initialGaussianStd;
        %             stepSize = 0.05;
        %             numberOfSteps  = 20;
        %             maximalIterations = 20;
        %             iterationResults = zeros(maximalIterations,2);
        %
        %             [~,currentThreshold] = obj.getFilteredFrame2(frameIndex, initialGaussianStd);
        %
        %
        %             while (maximalIterations > 0)
        %                 if (min(currentThreshold,targetThreshold) / max(currentThreshold,targetThreshold) > 0.9)
        %                     break;
        %                 end
        %
        %                 [~,ths] = arrayfun(@(gs)...
        %                     obj.getFilteredFrame2(frameIndex,gs),...
        %                     currentGaussianStd:stepSize:currentGaussianStd+numberOfSteps*stepSize, 'UniformOutput', false);
        %
        %                 ths = cell2mat(ths);
        %
        %                 diff = (targetThreshold - ths)';
        %
        %                 neighbordsDiffFit = fit((1:length(diff))', diff,'poly1');
        %                 if (neighbordsDiffFit.p1 <= 0)
        %                     currentGaussianStd = currentGaussianStd + stepSize * 2;
        %                 else
        %                     currentGaussianStd = currentGaussianStd - stepSize * 2;
        %                 end
        %
        %                 [~,currentThreshold] = obj.getFilteredFrame2(frameIndex, currentGaussianStd);
        %                 iterationResults(maximalIterations,:) = [currentGaussianStd abs(targetThreshold - currentThreshold)];
        %                 maximalIterations = maximalIterations - 1;
        %             end
        %
        %             if (maximalIterations == 0)
        %                  [~,ind] = min(iterationResults(:,2));
        %                 gaussianStd = iterationResults(ind,1);
        %             else
        %                 gaussianStd = currentGaussianStd;
        %             end
        %
        %         end
        
        
        function [gaussianStd] = modifyGaussianStd2(obj, frameIndex, initialGaussianStd, targetThreshold)
            
            [~,currentTh] = obj.getFilteredFrame2(frameIndex,initialGaussianStd);
            if ((min(currentTh,targetThreshold) / max(currentTh,targetThreshold)) > 0.9)
                gaussianStd = initialGaussianStd;
                return;
            end
            gaussianRange = initialGaussianStd:0.1:initialGaussianStd*2;
            
            [~,ths] = arrayfun(@(gs)...
                obj.getFilteredFrame2(frameIndex,gs),...
                gaussianRange, 'UniformOutput', false);
            
            
            
            ths = cell2mat(ths);
            
            d = abs(targetThreshold - ths);
            
            [~,ind] = min(d);
            gaussianStd = gaussianRange(ind);
        end
        
        function [frame,threshold] = getFilteredFrame2(obj, frameIndex, gaussianStd, threshold)
            logFilter = fspecial('log',round(gaussianStd * 4), gaussianStd);
            
            % Creating flat field.
            %firstFrame = obj.getRawFrame(1);
            currentFrame = obj.getRawFrame(frameIndex);
                        
            %flatFrame = min(255,currentFrame + (255 -firstFrame));
            
            filteredFrame = filter2(logFilter, currentFrame);
            
            if (~exist('threshold','var'))
                % Getting the optimized threshold
                threshold = graythresh(filteredFrame);
            end
            
            frame = im2bw(filteredFrame, threshold);
            % Debug - Remove the comment
            frame = ~frame;
        end
        
        function [frame,threshold] = getFilteredFrameRoberts2(obj, frameIndex, gaussianStd, threshold)
            logFilter = fspecial('log',round(gaussianStd * 4), gaussianStd);
            
            % Creating flat field.
            %firstFrame = obj.getRawFrame(1);
            currentFrame = obj.getRawFrame(frameIndex);
            %flatFrame = min(255,currentFrame + (255 -firstFrame));
            
            filteredFrame = filter2(logFilter, currentFrame);
            
            if (~exist('threshold','var'))
                % Getting the optimized threshold
                threshold = graythresh(filteredFrame);
            end
            
            frame = edge(filteredFrame, 'roberts');
            frame = ~frame;
        end
        
        
        
        % This method learn given a matrix of features, and save it's
        % learner in the obj.learner property.
        function [] = learn2(obj, allFeatures)
            
            %obj.learner.meanCentroid = mean(allFeatures,1);
            %obj.learner.threshold = quantile(pdist2(allFeatures, obj.learner.meanCentroid),0.95);
            
            % Scaling the featurse.
            maxCol = max(allFeatures);
            minCol = min(allFeatures);
            
            scaleFactor = repmat(maxCol - minCol,size(allFeatures,1),1);
            minMat = repmat(minCol, size(allFeatures,1) ,1);
            
            allFeatures = (allFeatures - minMat) ./ scaleFactor;
            % Fixing for division in Zero.
            allFeatures(isnan(allFeatures)) = 1;
            
            % Learning (Simple Distance).
            %             obj.learner.meanCentroid = mean(allFeatures,1);
            %             obj.learner.scaleFactor = maxCol - minCol;
            %             obj.learner.minValues = minCol;
            %             obj.learner.threshold = quantile(pdist2(allFeatures, obj.learner.meanCentroid),0.95);
            
            
            % Learning (Mahalanobis Distance)
            obj.learner.scaleFactor = maxCol - minCol;
            obj.learner.minValues = minCol;
            
            obj.learner.meanCentroid = mean(allFeatures,1);
            %DEBUG!! Uncomment
            obj.learner.covMat = eye(size(allFeatures,2)) .* cov(allFeatures);
            
            obj.learner.invCov = obj.learner.covMat / eye(length(obj.learner.covMat));
            
            % Calculating Mahalanobis distance for all samples.
            meanMat = repmat(obj.learner.meanCentroid, size(allFeatures,1), 1);
            distFromMean = allFeatures - meanMat;
            
            
            mDistances = distFromMean * obj.learner.invCov  * distFromMean';
            mDistances = diag(mDistances);
            mDistances = sqrt(mDistances);
            
            %
            obj.learner.threshold = quantile(mDistances,0.88);
        end
        
        % Classifying features of a new frame.
        function [classifiedCorrect] = classify2(obj,featuresToClassify)
            %classifiedCorrect = (pdist2(featuresToClassify, obj.learner.meanCentroid)) < obj.learner.threshold;
            
            
            % Scaling the featurse.
            minValues = repmat(obj.learner.minValues,size(featuresToClassify,1),1);
            scaleFactor = repmat(obj.learner.scaleFactor,size(featuresToClassify,1),1);
            
            
            featuresToClassify = (featuresToClassify - minValues) ./ scaleFactor;
            % Fixing for division in Zero.
            featuresToClassify(isnan(featuresToClassify)) = 1;
            
            %classifiedCorrect = (pdist2(featuresToClassify, obj.learner.meanCentroid)) < obj.learner.threshold;
            
            % Mahalanobis Distance
            meanMat = repmat(obj.learner.meanCentroid, size(featuresToClassify,1), 1);
            distanceFromMean = featuresToClassify - meanMat;
            
            mDistances = distanceFromMean * obj.learner.invCov * distanceFromMean';
            mDistances = diag(mDistances);
            mDistances = sqrt(mDistances);
            
            classifiedCorrect = mDistances < obj.learner.threshold;
        end
        
        
        function [centroids, features,extermas,idxs] = extractFeatures2(obj,frameIndex, gaussianStd)
            
            % We extract features both from the filtered and the raw image.
            rawFrame = obj.getRawFrame(frameIndex);
            
            filteredFrame = obj.getFilteredFrame2(frameIndex,gaussianStd);
            filteredFrameRoberts = obj.getFilteredFrameRoberts2(frameIndex, gaussianStd);
            
            numOfFilteredFeatures = 2;
            numOfRawFeatures = 3;
            
            % Filtered Frame features.
            %regProps = regionprops(~filteredFrame,{'Centroid','Area','Solidity','BoundingBox'});
            regProps = regionprops(~filteredFrame,{'Centroid','Area','PixelIdxList','Extrema'});
            features= zeros(length(regProps), numOfFilteredFeatures + numOfRawFeatures);
            extermas = cell(length(regProps),1);
            idxs = cell(length(regProps),1);
            centroids = zeros(length(regProps),2);
            
            
            % Orginizing and setting the raw features
            for i=1:length(regProps)
                % Getting Centroids
                centroids(i,:) = regProps(i).Centroid;
                
                % Filtered features.
                features(i,1) = regProps(i).Area;
                features(i,2) = sum(filteredFrameRoberts(regProps(i).PixelIdxList)) / length(regProps(i).PixelIdxList);
                
                % Raw Features
                features(i,3) = median(double(rawFrame(regProps(i).PixelIdxList)));
                features(i,4) = min(double(rawFrame(regProps(i).PixelIdxList)));
                features(i,5) = max(double(rawFrame(regProps(i).PixelIdxList)));
                
                if (isnan(features(i,5)))
                    features(i,5) = features(i,4);
                end
                
                
                
                
                % Saving idxs
                extermas{i} = regProps(i).Extrema;
                idxs{i} = regProps(i).PixelIdxList;
            end
            
            % Now we're scaling the features.
            %maxCol = max(features);
            %minCol = min(features);
            
            %scaleFactor = repmat(maxCol - minCol,size(features,1),1);
            %minMat = repmat(minCol, size(features,1) ,1);
            
            %features = (features - minMat) ./ scaleFactor;
        end
        
        function [features] = uiCreateTraining2(obj, frameIndex, gaussianStd)
            % We extract features both from the filtered and the raw image.
            filteredFrame = obj.getFilteredFrame2(frameIndex,gaussianStd);
            [~,features,extermas,idxs] = obj.extractFeatures2(frameIndex, gaussianStd);
            labels = zeros(size(features,1), 1);
            
            % Duplicate Frame for RGB
            rgbFrame = double(repmat(filteredFrame, [1 1 3]));
            
            
            % Show frame.
            imshow(rgbFrame);
            title('Right click to exit input mode');
            hold on;
            
            goOnChoice = 1;
            while(goOnChoice)
                [x,y,button] = ginput(1);
                
                if (button ~= 1)
                    goOnChoice = 0;
                    continue;
                end
                
                % Getting the rounded coordiantes.
                x = ceil(x);
                y = ceil(y);
                
                % Looking for figure that matches that choice.
                matchingRegions = cellfun(@(exterma) inpolygon(x,y,exterma(:,1), exterma(:,2)), extermas);
                
                if (any(matchingRegions))
                    greenFrame = rgbFrame(:,:,2);
                    greenFrame(idxs{matchingRegions}) = 1;
                    %greenFrame = roifill(greenFrame, selectedExterma(:,1), selectedExterma(:,2));
                    rgbFrame(:,:,2) = greenFrame;
                    
                    imshow(rgbFrame);
                    
                    labels(matchingRegions) = 1;
                end
            end
            
            features = features(logical(labels),:);
        end
        
        function []=uiSelectWorms2(obj,frameIndex,gaussianStd, features, tolerence)
            % We extract features both from the filtered and the raw image.
            filteredFrame = obj.getFilteredFrame2(frameIndex,gaussianStd);
            [~,frameFeatures,~,idxs] = obj.extractFeatures2(frameIndex, gaussianStd);
            
            % Duplicate Frame for RGB
            rgbFrame = double(repmat(filteredFrame, [1 1 3]));
            
            
            % "Learning" from features;
            featuresMeans = mean(features);
            featuresCorrMat = corr(features);
            
            % Deciding the right threshold.
            threshold = mvnpdf(featuresMeans, featuresMeans, featuresCorrMat) * tolerence;
            
            positiveLabels = obj.selectObjects2(frameFeatures,featuresMeans, featuresCorrMat, threshold);
            
            redFrame = rgbFrame(:,:,1);
            positiveShapes = find(positiveLabels');
            for i=positiveShapes
                currentIdxs = idxs{i};
                
                redFrame(currentIdxs) = 1;
            end
            
            rgbFrame(:,:,1) = redFrame;
            imshow(rgbFrame);
        end
        
        
        function [positiveLabels] = selectObjects2(obj, frameFeatures, means,corrMat, threshold)
            positiveLabels = mvnpdf(frameFeatures, means, corrMat) > threshold;
        end
        
        
        
        function frame = getFilteredFrameSeq(obj, frameIndex, binaryThreshold)
            if (nargin < 3)
                binaryThreshold = 0.98;
            end
            
            % Looking at prevFrames last frames.
            prevFramesNum = 5;
            
            frame = obj.getRawFrame(frameIndex);
            flatMat = [];
            
            % Preparing the previous frames for next frame.
            if ~isempty(obj.prevFrames)
                storedFrames = size(obj.prevFrames,3);
            else
                storedFrames = 0;
                
            end
            
            if (storedFrames > 0)
                flatMat = uint8(mean(obj.prevFrames,3));
            end
            
            
            % Preparing the stored frames array for the next sequential
            % requset.
            if (storedFrames < prevFramesNum)
                obj.prevFrames(:,:,storedFrames + 1) = frame;
            else
                obj.prevFrames(:,:,prevFramesNum + 1) = frame;
                obj.prevFrames = obj.prevFrames(:,:,2:prevFramesNum+1);
            end
            
            % Collect flat mat matrix
            if (~isempty(flatMat))
                % Running filter
                frame = obj.applyFlatField(frame, flatMat);
            end
            
            frame = obj.applyContrast(frame,binaryThreshold);
        end
        
        
        function [] = resetPrevFrames(obj)
            obj.prevFrames = uint8([]);
        end
        
        
        % Applies flat field to the video.
        function decMatrix = initializeFlatField(obj)
            
            % Getting temp frame.
            firstFrame = obj.getRawFrame(1);
            
            height = size(firstFrame,1);
            width = size(firstFrame,2);
            colors = size(firstFrame,3);
            
            % Decrement Matrix.
            %decMatrix = uint8(zeros(height,width, colors));
            
            %for i=1:obj.numberOfFrames
            %for i=(obj.numberOfFrames-20):obj.numberOfFrames
            %    currentFrame = obj.getRawFrame(i);
            %    decMatrix = max(currentFrame,decMatrix);
            %end
            
            decMatrix = obj.getRawFrame(1);
        end
        
        % Apply contrast
        function currentFrame = applyFlatField(obj, currentFrame, flatFieldMat)
            %currentFrame = min(255,uint8(currentFrame + (255 - flatFieldMat)));
            currentFrame = wiener2(min(255,uint8(currentFrame + (255 - flatFieldMat))),[5 5]);
        end
        
        % Applies contrast to the movie.
        function currentFrame = applyContrast(obj, currentFrame, binaryThreshold)
            frameMat = currentFrame(:,:,1);
            
            % threshold = graythresh(frameMat);
            frameMat = im2bw(frameMat, binaryThreshold);
            
            if (size(currentFrame,3) > 1)
                currentFrame(:,:,1) = frameMat * 255;
                currentFrame(:,:,2) = frameMat * 255;
                currentFrame(:,:,3) = frameMat * 255;
            else
                currentFrame = frameMat * 255;
            end
        end
        
        
        
        function [predictedTrack] = predictTrack(obj,track)
            newState  = [1 0 1 0; 0 1 0 1;0 0 1 0; 0 0 0 1] * track.state;
            currentCentroid = newState(1:2);
            currentProperties = zeros(1,size(track.properties,2));
            stepSize = pdist2(track.state(1:2)', currentCentroid');
            predictedTrack = track;
            
            
            predictedTrack.lastCoordinates =  currentCentroid';
            predictedTrack.properties(track.numberOfSteps + 1,:) = currentProperties;
            predictedTrack.lastStepSize = stepSize;
            predictedTrack.stepSizes(track.numberOfSteps + 1) =  stepSize;
            % Adding a predicted step.
            predictedTrack.path(track.numberOfSteps + 1,:) = [(track.path(track.numberOfSteps,1) + 1) currentCentroid' 1];
            predictedTrack.numberOfSteps = predictedTrack.numberOfSteps + 1;
            
            predictedTrack.state = newState;
        end
        
        % DEBUG
        function  [modifiedTracks] = extractMoreFeaturesForTracks(obj,tracks, trainingFeatures, frameThresholds)
            % Learning the training features.
            obj.learn2(trainingFeatures);
            
            for i=1:obj.numberOfFrames
                % Frame entites.
                [centroids,frameProperties,~,~] = obj.extractFeatures2(i, frameThresholds(i));
                
                % Classifying the found entites.
                acceptingIndices = obj.classify2(frameProperties);
                
                % Getting the relevant centroids
                relevantCentroids = centroids(acceptingIndices,:);
                
                % Frame new features.
                % We extract features both from the filtered and the raw image.
                filteredFrame = obj.getFilteredFrame2(i,frameThresholds(i));
                
                % Filtered Frame features.
                regProps = regionprops(~filteredFrame,{'MajorAxisLength','MinorAxisLength'});
                
                relevantRegProps = regProps(acceptingIndices);
                
                for j=1:length(tracks)
                    t = tracks(j);
                    
                    if (t.path(1,1) > i || t.path(end,1) < i)
                        continue
                    end
                    
                    frameIndex = (t.path(:,1) == i);
                    
                    if (~any(frameIndex))
                        error('Something wrong with tracks frame numbering.');
                        continue;
                    end
                    
                    frameEntitesIndices =sub2ind(fliplr(size(filteredFrame)),relevantCentroids(:,1), relevantCentroids(:,2));
                    currentPosInd = sub2ind(fliplr(size(filteredFrame)), t.path(frameIndex,2), t.path(frameIndex,3));
                    
                    currentFrameInEntitesInd = (frameEntitesIndices == currentPosInd);
                    
                    if (~any(currentFrameInEntitesInd))
                        if (t.path(frameIndex,4) == 1)
                            tracks(j).path(frameIndex,5) = -1;
                            tracks(j).path(frameIndex,6) = -1;
                        else
                            error('Something went wrong. Check your code');
                        end
                    else
                        tracks(j).path(frameIndex,5) = relevantRegProps(currentFrameInEntitesInd).MajorAxisLength;
                        tracks(j).path(frameIndex,6) = relevantRegProps(currentFrameInEntitesInd).MinorAxisLength;
                    end
                end
                
                disp(['Processed Frame ' num2str(i)]);
            end
            
            modifiedTracks = tracks;
        end
        
        %
        
        function [tracks,tracksStats] = performTracking3(obj,trainingFeatures, frameThresholds, imgH)
            
            % Learning the training features.
            obj.learn2(trainingFeatures);

            % Initializing the empty tracks.
            obj.EMPTY_TRACK  = struct('lastCoordinates',[],...
                'properties',zeros(obj.numberOfFrames,size(trainingFeatures,2)),...
                'lastStepSize',[],...
                'stepSizes',zeros(1,obj.numberOfFrames),...
                'path',zeros(obj.numberOfFrames,4),...
                'id',double(0), ...
                'interesting',double(0), ...
                'isActive',false,...
                'isUsed',0,...
                'state', zeros(4,1),...
                'previousState', zeros(4,1),...
                'predicted',0,...
                'numberOfSteps', double(0));
            obj.tracks = repmat(obj.EMPTY_TRACK,8000,1);
            
            % Counter for tracks.
            numberOfTracks = 0;
            
            % Savign statistics regarding the tracking process.
            tracksStats = zeros(obj.numberOfFrames,2);
            
            % Running over all of the frames
            for i=1:obj.numberOfFrames
            %for i=45000:50500
            %for i=1:1440
            % for i=1:150
                %for i=1:1000
                % A matrix with centroid position and area.
                [centroids,frameProperties,extermas,idxs] = obj.extractFeatures2(i, frameThresholds(i));
                
                % Active tracks
                activeTracks = obj.tracks([obj.tracks.isActive] == 1);
                
                if (isempty(centroids))
                    % Closign all active tracks.
                    for t=1:length(activeTracks)
                        activeTracks(t).isActive = 0;
                    end
                end
                
                % Accepting entites. Classifying entites given the training
                % set.
                if (obj.SHOULD_ML)
                    acceptingIndices = obj.classify2(frameProperties);
                else
                    acceptingIndices = logical(ones(1, size(frameProperties,1)));
                end
                
                % Saving stats about the tracking process.
                tracksStats(i,:) = [size(frameProperties,1),sum(acceptingIndices)];
                
                relevantCentroids = centroids(acceptingIndices,:);
                relevantProperties = frameProperties(acceptingIndices,:);

                
                % Here we write which tracks are assigned to each entity.
                entityRegistered = cell(size(relevantProperties,1),1);
                
                % Going over the active tracks.
                for j=1:length(activeTracks)
                    %Removing exceedingly long predicted tracks
                    if (obj.SHOULD_PREDICT && activeTracks(j).predicted > obj.ALLOW_PRED_STEPS)
                        activeTracks(j).isActive = 0;
                        obj.tracks(activeTracks(j).id) = activeTracks(j);
                        continue;
                    end
                    
                    % Getting the last position of the current track.
                    currentTrackPosition = activeTracks(j).lastCoordinates;
                    
                    % Looking for close relevant entries.
                    distances = pdist2(currentTrackPosition, relevantCentroids);
                    
                    % Getting the closest entity.
                    [minDist, entityId] = min(distances);
                    
                    %if (isempty(entityId))
                    %    continue;
                    %end
                    
                    % Calculate step probability.
                    stepProbability = obj.getStepProbability(activeTracks(j), minDist);
                    
                    if (isempty(minDist) || (stepProbability < eps) || (isempty(relevantProperties)))
                    %if (isempty(minDist) || (minDist > maximalBodySize * stepFactor) || (isempty(relevantProperties)))
                    %if (isempty(minDist)  || (isempty(relevantProperties)))
                        % Step is too big - trying to predict step.
                        if (obj.SHOULD_PREDICT && ...
                            any(activeTracks(j).state ~= 0) && ...
                            activeTracks(j).numberOfSteps > 2 && ...
                            (activeTracks(j).predicted <= obj.ALLOW_PRED_STEPS))
                            
                            activeTracks(j) = obj.predictTrack(activeTracks(j));
                            activeTracks(j).predicted = activeTracks(j).predicted + 1;
                        else
                            activeTracks(j).isActive = 0;
                        end
                        
                        obj.tracks(activeTracks(j).id) = activeTracks(j);
                        continue;
                    end
                    
                    % Writing the new step
                    activeTracks(j).lastCoordinates =  relevantCentroids(entityId,:);
                    activeTracks(j).properties(obj.tracks(trackId).numberOfSteps + 1,:) = relevantProperties(entityId,:);
                    activeTracks(j).lastStepSize = minDist;
                    activeTracks(j).stepSizes(activeTracks(j).numberOfSteps) =  minDist;
                    activeTracks(j).path(activeTracks(j).numberOfSteps + 1,:) = [i relevantCentroids(entityId,:) 0];
                    %activeTracks(j).predicted = 0;
                    
                    
                    % Increasing step numbers.
                    activeTracks(j).numberOfSteps = activeTracks(j).numberOfSteps + 1;

                    % Updating the state
                    activeTracks(j).previousState = activeTracks(j).state;
                    
                    activeTracks(j).state = zeros(4,1);
                    activeTracks(j).state(1:2) = relevantCentroids(entityId,:);
                    % Calculating the veolocity for the state.
                    if (activeTracks(j).numberOfSteps > 1)
                        lastVector = activeTracks(j).path(activeTracks(j).numberOfSteps,2:3) ...
                            - activeTracks(j).path((activeTracks(j).numberOfSteps-1),2:3);

                        % Vy
                        activeTracks(j).state(3) = dot([1 0], lastVector);

                        % Vx
                        activeTracks(j).state(4) = dot([0 1], lastVector);
                    end
                                        
                    % Saving which track was assigned to which entity. To
                    % solve amibiguities.
                    entityRegistered{entityId} = [entityRegistered{entityId} activeTracks(j).id];
                    
                    % Saving the active track.
                    obj.tracks(activeTracks(j).id) = activeTracks(j);
                end
                
                % Cleaning unused memory
                if (mod(i,20) == 0)
                    disp('Cleaning unused allocated memory..');
                    
                    % Getting all inactive used tracks.
                    inactiveTracks = obj.tracks(intersect(find([obj.tracks.isActive] == false),...
                        find([obj.tracks.isUsed] == 1)));
                    
                    for t=1:length(inactiveTracks)
                        currentTrack = inactiveTracks(t);
                        currentTrack = obj.cleanTrack(currentTrack);
                        obj.tracks(currentTrack.id) = currentTrack;
                    end
                end
                
                
                
                %Solve Amibiguities
                obj.solveAmbiguities(entityRegistered);
                
                % Going over unregistered entitoes.
                unassignedEntitesIndices = cellfun(@(entityReg) isempty(entityReg), entityRegistered);
                unassigedCentroids = relevantCentroids(unassignedEntitesIndices,:);
                unassignedProperties = relevantProperties(unassignedEntitesIndices,:);
                
                % Going over unassigned properites.
                for t=1:size(unassigedCentroids,1)
                    numberOfTracks = numberOfTracks + 1;
                    trackId = numberOfTracks;
                    
                    obj.tracks(trackId).lastCoordinates = unassigedCentroids(t,1:2);
                    obj.tracks(trackId).properties(1,:) = unassignedProperties(t,:);
                    obj.tracks(trackId).lastStepSize = 0;
                    obj.tracks(trackId).path(1,:)  = [i unassigedCentroids(t,1:2) 0];
                    obj.tracks(trackId).id = trackId;
                    obj.tracks(trackId).interesting = 0;
                    obj.tracks(trackId).isActive = 1;                    obj.tracks(trackId).numberOfSteps = 1;
                    obj.tracks(trackId).isUsed = 1;
                    obj.tracks(trackId).predicted = 0;
                end
                
                % Logging
                disp(['Movie Name: ' obj.name ...
                    ' Processing frame: ' num2str(i) ...`
                    ' Active tracks:' num2str(length(activeTracks)) ...
                    ' New tracks:' num2str(size(unassigedCentroids,1)) ...
                    ' All Tracks:' num2str(numberOfTracks) ...
                    ' Relevant Entities: ' num2str(sum(acceptingIndices)), ...
                    ' Frame Entities:' num2str(length(acceptingIndices))]);
                
            end
            
            % Clear all empty tracks.
            obj.tracks([obj.tracks.id] == 0) = [];
            
            
            for i=1:length(obj.tracks)
                % Clear empty allocation within tracks.
                % Cleaning the track.
                currentTrack = obj.tracks(i);
                currentTrack = obj.cleanTrack(currentTrack);
                obj.setTrackById(currentTrack.id, currentTrack);
                                
                % Removing predicted suffixes.
                predictedVector = obj.tracks(i).path(:,4);
                predictedVector = flipud(predictedVector);
                
                firstNotPredicted = find(predictedVector == 0,1,'first');
                if (firstNotPredicted ~= 1)
                    obj.tracks(i).path = obj.tracks(i).path(1:(end - firstNotPredicted + 1),:);
                    obj.tracks(i).numberOfSteps = obj.tracks(i).numberOfSteps - firstNotPredicted + 1;
                end
            end
            
            %Tracks filtering
            disp('Filtering tracks.');
            
            % No tracks to filter.
            if (numberOfTracks == 0)
                return;
            end
            
            % Clear unimportant tracks.
            %pathPolyArea = arrayfun(@(t) polyarea(t.path(:,2), t.path(:,3)), obj.tracks);
            
            % Removing 2 points (and less) tracks.
            % DEBUG - UNCOMMENT!!!!!!
            obj.tracks(arrayfun(@(t) size(t.path,1), obj.tracks) <=2) = [];
            
            % Prepare steps sizes
            stepSizes = cell(1, length(obj.tracks));
            numberOfTracks = length(obj.tracks);
            
            for i=1:numberOfTracks
                currentPath = obj.tracks(i).path(:,2:3);
                currentSteps = diff(currentPath);
                stepSizes{i} = sqrt(currentSteps(:,1).^2 + currentSteps(:,2).^2);
            end
            
            stepSizes = cell2mat(stepSizes');
            stepThreshold = quantile(stepSizes, 0.997);
            
            tracksToRemove = zeros(1,length(obj.tracks));
            splittedTracks = [];
            lastId = obj.tracks(end).id;
            
            for i=1:numberOfTracks
                currrentTrack = obj.tracks(i);
                
                newSplittedTracks = obj.splitTrack(currrentTrack, stepThreshold, lastId);
                
                if (~isempty(newSplittedTracks))
                    
                    splittedTracks = [splittedTracks; newSplittedTracks];
                    tracksToRemove(i) = 1;
                    lastId = splittedTracks(end).id;
                end
            end
            
            obj.tracks(logical(tracksToRemove)) = [];
            obj.tracks = [obj.tracks;splittedTracks];
            
            
            for i=1:numberOfTracks
                % Setting the Chemoattractant cords.
                obj.tracks(i).chemoCords = obj.chemoCords;
                
                % Saving the parent object.
                obj.tracks(i).vt = obj;
                
                % Saving the name of the video
                obj.tracks(i).name = obj.name;
                
                
                % Returning the created tracks.
                tracks = obj.tracks;
            end
            
            % Removing fuzzy initial phase of tracks.
%             for i=1:length(obj.tracks)
%                 currentPath = obj.tracks(i).path(:,2:3);
%                 
%                 currentSteps = diff(currentPath);
%                 currentStepsSizes = sqrt(currentSteps(:,1).^2 + currentSteps(:,2).^2);
%                 
%                 if (length(currentSteps) < obj.LEARNING_STEPS)
%                     continue;
%                 end
%                 
%                 % Getting rid of still steps.
%                 currentStepsSizes = currentStepsSizes(currentStepsSizes ~= 0);
%                 
%                 initialPhase = currentStepsSizes(1:obj.LEARNING_STEPS);
%                 secondaryPhase = currentStepsSizes(obj.LEARNING_STEPS:length(currentStepsSizes));
%                                 
%                 [~,p] = ttest2(initialPhase, secondaryPhase,'tail','right');
%                 
%                 if (p < 10e-6)
%                     obj.tracks(i).path = obj.tracks(i).path((obj.LEARNING_STEPS + 1):end,:);
%                     obj.tracks(i).numberOfSteps = obj.tracks(i).numberOfSteps - obj.LEARNING_STEPS;
%                 elseif any(find(currentStepsSizes > stepThreshold))
%                     startPos = find(currentStepsSizes > stepThreshold,1,'last');
%                     endPos = size(currentPath,1);
%                     
%                     obj.tracks(i).path = obj.tracks(i).path(startPos:endPos,:);
%                     obj.tracks(i).numberOfSteps = endPos - startPos;
%                     
%                 end
% 
%             end
            
        end
        
        function splittedTracks = splitTrack(obj, trackToSplit, threshold, lastId)
            currentPath = trackToSplit.path(:,2:3);
            steps = diff(currentPath);
            stepSizes = sqrt(steps(:,1).^2 + steps(:,2).^2);
            stepSizes = stepSizes(:);
            
            weirdSteps = find(stepSizes > threshold) + 1;
            
            if (isempty(weirdSteps))
                splittedTracks = [];
                return;
            end
            
            splitPoses = [1; weirdSteps ;size(currentPath,1)];
            splitPoses = min(splitPoses, trackToSplit.numberOfSteps);
            
            currentPos = 1;
            currentId = lastId + 1;
            count = 1;
            
            splittedTracks = repmat(obj.EMPTY_TRACK,length(splitPoses) - 1,1);
            while (currentPos ~= splitPoses(end))
                
                if ((splitPoses(count + 1) - currentPos - 1) > 0)
                    splittedTracks(count).id = currentId;
                    splittedTracks(count).path = trackToSplit.path(((currentPos)+1):(splitPoses(count + 1)-1),:);
                    splittedTracks(count).numberOfSteps = splitPoses(count + 1) - currentPos  - 1;
                end
                currentId = currentId  + 1;
                count = count + 1;
                currentPos = splitPoses(count);
            end
            
            splittedTracks = splittedTracks([splittedTracks.numberOfSteps] > 0);
        end
        
        % Process Tracking - Alternate function
        function performTracking2(obj, wormProperties, showProgress, gaussianStd, tolerance)
            
            % Create Expectation and variance of worm properties.
            numberOfStatists = size(wormProperties,2);
            propertiesMeans = mean(wormProperties);
            propertiesCov = cov(wormProperties) + 0.001 * eye(size(wormProperties,2));
            
            % Initialize the tracks
            emptyTrack = struct('lastCoordinates',[],...
                'properties',zeros(obj.numberOfFrames,numberOfStatists),...
                'lastStepSize',[],...
                'stepSizes',zeros(1,obj.numberOfFrames),...
                'path',zeros(obj.numberOfFrames,3),...
                'id',double(0), ...
                'interesting',double(0), ...
                'isActive',false,...
                'isUsed',0,...
                'numberOfSteps', double(0));
            
            obj.tracks = repmat(emptyTrack,80000,1);
            numberOfTracks = 0;
            
            % Restting previous frames.
            obj.resetPrevFrames();
            
            for i=1:obj.numberOfFrames
            %for i=770:obj.numberOfFrames
                %for i=738:750
                currentGaussianStd = obj.modifyGaussianStd2(i, gaussianStd, 0.4);
                
                % A matrix with centroid position and area.
                [centroids,properties,~,~] = obj.extractFeatures2(i, currentGaussianStd);
                
                % Couldn't find anything in the image.
                if isempty(centroids)
                    activeTracks = obj.tracks([obj.tracks.isActive] == 1);
                    
                    % Closing all active tracks.
                    for j=1:length(activeTracks)
                        obj.tracks(activeTracks(j).id).isActive = 0;
                    end
                    continue;
                end
                
                
                
                % Creating the cetroids track "register" matrix.
                % Creating addtinal colum for track regitration.
                frameEntites = zeros(size(centroids,1),size(centroids,2) + size(wormProperties,2) + 1); % Two columns for centroid, one for area, and one for registration.
                frameEntites(:,1:2) = centroids;
                frameEntites(:,3:2+numberOfStatists) = properties;
                
                positiveEntities = obj.selectObjects2(frameEntites(:,3:2+numberOfStatists), propertiesMeans, propertiesCov, tolerance);
                
                %Getting the relevant entities
                relevantEntities = frameEntites(positiveEntities,:);
                
                % Here we write which tracks are assigned to each entity.
                trackRegistered = cell(size(relevantEntities,1),1);
                
                % Add entities to tracks
                % Going over the tracks and trying to register an entity to a
                % track.
                if (~isempty(obj.tracks))
                    activeTracks = obj.tracks([obj.tracks.isActive] == 1);
                else
                    activeTracks = [];
                end
                
                for j=1:length(activeTracks)
                    % If we don't have more entities left.
                    if isempty(relevantEntities)
                        activeTracks(j).isActive = false;
                    else
                        currentTrackPosition = activeTracks(j).lastCoordinates;
                        
                        distances = pdist2(currentTrackPosition, relevantEntities(:,1:2));
                        [minDist, entityId] = min(distances);
                        
                        if (minDist > 50)
                            activeTracks(j).isActive = 0;
                        else
                            activeTracks(j).lastCoordinates =  relevantEntities(entityId,1:2);
                            activeTracks(j).properties(obj.tracks(trackId).numberOfSteps + 1,:) = relevantEntities(entityId,3:2+numberOfStatists);
                            activeTracks(j).lastStepSize = minDist;
                            activeTracks(j).stepSizes(activeTracks(j).numberOfSteps) =  minDist;
                            activeTracks(j).path(activeTracks(j).numberOfSteps + 1,:) = [i relevantEntities(entityId,1:2)];
                            
                            % Marking that this entity has already been assigned to a
                            % track.
                            relevantEntities(entityId,end) = 1;
                            
                            % Saving which track was assigned to which entity. To
                            % solve amibiguities.
                            trackRegistered{entityId} = [trackRegistered{entityId} activeTracks(j).id];
                            
                            % Check if this track is interesting.
                            if (activeTracks(j).numberOfSteps > 10)
                                if mean((activeTracks(j).stepSizes(1:activeTracks(j).numberOfSteps)) >=10)
                                    activeTracks(j).interesting = 1;
                                end
                            end
                        end
                        
                        % Increasing step numbers.
                        activeTracks(j).numberOfSteps = activeTracks(j).numberOfSteps + 1;
                    end
                    
                    % Saving the current track.
                    obj.setTrackById(activeTracks(j).id, activeTracks(j));
                end
                
                % Cleaning unused memory
                if (mod(i,20) == 0)
                    disp('Cleaning unused allocated memory..');
                    
                    % Getting all inactive used tracks.
                    inactiveTracks = obj.tracks(intersect(find([obj.tracks.isActive] == false),...
                        find([obj.tracks.isUsed] == 1)));
                    
                    for t=1:length(inactiveTracks)
                        currentTrack = inactiveTracks(t);
                        currentTrack = obj.cleanTrack(currentTrack);
                        obj.setTrackById(currentTrack.id, currentTrack);
                    end
                end
                
                
                
                % Create new tracks
                % Getting the unregistered entities.
                unregisteredEntities =[];
                if (~isempty(relevantEntities))
                    unregisteredEntities = relevantEntities(relevantEntities(:,end) == 0,:);
                end
                
                for j=1:size(unregisteredEntities,1)
                    trackId = numberOfTracks + 1;
                    numberOfTracks = numberOfTracks + 1;
                    
                    obj.tracks(trackId).lastCoordinates = unregisteredEntities(j,1:2);
                    obj.tracks(trackId).properties(1,:) = unregisteredEntities(j,3:2+numberOfStatists);
                    obj.tracks(trackId).lastStepSize = 0;
                    obj.tracks(trackId).path(1,:)  = [i unregisteredEntities(j,1:2)];
                    obj.tracks(trackId).id = trackId;
                    obj.tracks(trackId).interesting = 0;
                    obj.tracks(trackId).isActive = 1;
                    obj.tracks(trackId).numberOfSteps = 1;
                    obj.tracks(trackId).isUsed = 1;
                end
                
                %Solve Amibiguities
                obj.solveAmbiguities(trackRegistered);
                
                
                disp(['Movie Name: ' obj.name ...
                    ' Processing frame: ' num2str(i) ...`
                    ' Active tracks:' num2str(length(activeTracks)) ...
                    ' New tracks:' num2str(size(unregisteredEntities,1)) ...
                    ' All Tracks:' num2str(numberOfTracks) ...
                    ' Relevant Entities: ' num2str(length(relevantEntities)), ...
                    ' Frame Entities:' num2str(length(frameEntites))]);
                
                if (showProgress == 1)
                    % Show frame
                    obj.viewActiveTracksOnFrame(i,obj.tracks);
                end
            end
            
            % Clear all empty tracks.
            obj.tracks([obj.tracks.id] == 0) = [];
            
            % Remove static routes (probably noise)
            tracksAreas = arrayfun(@(t) (max(t.path(:,3) - min(t.path(:,3)))) * (max(t.path(:,2) - min(t.path(:,2)))), ...
                obj.tracks);
            
            % We look for tracks which span an area less then 1000.
            smallTracksIds = find(tracksAreas <= 650);
            
            for i=1:length(smallTracksIds)
                obj.tracks(smallTracksIds(i)).interesting = 0;
            end
            
            % Clear empty allocation within tracks.
            for i=1:length(obj.tracks)
                % Cleaning the track.
                currentTrack = obj.tracks(i);
                currentTrack = obj.cleanTrack(currentTrack);
                obj.setTrackById(currentTrack.id, currentTrack);
                
                % Setting the Chemoattractant cords.
                obj.tracks(i).chemoCords = obj.chemoCords;
                
                % Saving the parent object.
                obj.tracks(i).vt = obj;
                
                % Saving the name of the video
                obj.tracks(i).name = obj.name;
            end
        end
        
        function probability = getStepProbability(obj, track, newStep)
            MINIMAL_TRJ_SIZE = 5;
            
            if (track.numberOfSteps < MINIMAL_TRJ_SIZE)
                probability = 1;
            else    
                path = track.path(1:track.numberOfSteps,2:3);
            
                diffPath = diff(path);
                stepSizes = sqrt(diffPath(:,1).^2 + diffPath(:,2).^2);
            
                nonZeroStepSizes = stepSizes(stepSizes ~= 0);
                probability = normpdf(newStep, mean(nonZeroStepSizes), std(nonZeroStepSizes));
            end
        end
        
        % Return tracks by their id.
        function tracks = getTracksById(obj,ids)
            %trackIndices = arrayfun(@(track) sum(ids == track.id) > 0, obj.tracks);
            tracks = obj.tracks(ids);
        end
        
        % Set track by its id.
        function setTrackById(obj, id, newTrack)
            %obj.tracks([obj.tracks.id] == id) = newTrack;
            obj.tracks(id) = newTrack;
        end
        
        
        % Solving ambiguities between two tracks assigned for the same
        % entity
        function solveAmbiguities(obj,entityRegistration)
            
            entityRegistration = entityRegistration(~cellfun(@isempty,entityRegistration));
            
            for i=1:length(entityRegistration)
                % Current Ambiguis tracks
                ambiguisTracksIds = entityRegistration{i};
                
                % Checking for entities with more than one tracks registered.
                if (length(ambiguisTracksIds) == 1)
                    currentTrack = obj.getTracksById(ambiguisTracksIds);
                    currentTrack.predicted = 0;
                    obj.setTrackById(currentTrack.id, currentTrack);
                    continue;
                end
                                
                % Getting the ambiguis tracks
                ambTracks = obj.getTracksById(ambiguisTracksIds);
                
                % Getting rid of small tracks.
                % Here we store the tracks we will eventually close.
                % Short tracks, predicted tracks.
                tracksToStopIds = zeros(1,length(ambTracks));
                
                % First deletion: small tracks.
                tracksToStopIds([ambTracks.numberOfSteps] <= obj.LEARNING_STEPS) = 1;
                %tracksToStopIds([ambTracks.predicted] > 2) = 1;
                
                
                %tracksToStop = ambTracks(logical(tracksToStopIds));
                
                %for trackToStop=tracksToStop'
                %    trackToStop.path(trackToStop.numberOfSteps,:) = [0 0 0 0];
                %    trackToStop.stepSizes(trackToStop.numberOfSteps) = 0;
                %    trackToStop.numberOfSteps = trackToStop.numberOfSteps - 1;
                %    trackToStop.isActive = 0;
                    
                %    obj.setTrackById(trackToStop.id, trackToStop);
                %end
                
                % Getting rid of tracks we've stopped.
                %ambTracks(logical(tracksToStopIds)) = [];
                                
                
                % Second del: Getting rid of weird steps.
                % Here we store the tracks we will eventually close.
                % Short tracks, predicted tracks.                
                probVals = zeros(1,length(ambTracks));
                for j=1:length(ambTracks)
                    
                    % if the track was already scheduled to stop.
                    if (tracksToStopIds(j) == 1)
                        continue;
                    end
                    
                    t = ambTracks(j);
                    currentPath = t.path(1:t.numberOfSteps,2:3);
                    steps = diff(currentPath);
                    stepSizes = sqrt(steps(:,1).^2 + steps(:,2).^2);
                    lastStep = stepSizes(end);
                    
                    tempTrack = t;
                    tempTrack.path = tempTrack.path;
                    tempTrack.path(t.numberOfSteps,:) = 0;
                    tempTrack.numberOfSteps = tempTrack.numberOfSteps - 1;
                    
                    probVals(j) = obj.getStepProbability(tempTrack, lastStep);
                end
                
                tracksToStopIds = tracksToStopIds | (probVals < (max(probVals) / 10));
                
                
                % Taking care of "deleted" tracks.
                tracksToStop = ambTracks(tracksToStopIds);
                
                for trackToStop=tracksToStop'
                    trackToStop.path(trackToStop.numberOfSteps,:) = [0 0 0 0];
                    
                    % Trimming.
                    trackToStop.stepSizes(trackToStop.numberOfSteps) = 0;
                    trackToStop.numberOfSteps = trackToStop.numberOfSteps - 1;
                    trackToStop.state = trackToStop.previousState;
                    
                    if (obj.SHOULD_PREDICT && any(trackToStop.state ~= 0) && trackToStop.predicted == 0)
                        trackToStop = obj.predictTrack(trackToStop);
                        trackToStop.predicted = trackToStop.predicted + 1;
                    else
                        trackToStop.isActive = 0;
                    end
                    
                    obj.setTrackById(trackToStop.id, trackToStop);
                end
                
                % The unresolve ambigouis tracks.
                ambTracks = ambTracks(~tracksToStopIds);
                
                if (length(ambTracks) ~= 1)                
                    for trackToPredict=ambTracks'    
                        % Trimming.
                        trackToPredict.path(trackToPredict.numberOfSteps,:) = [0 0 0 0];
                        trackToPredict.numberOfSteps = trackToPredict.numberOfSteps - 1;
                        trackToPredict.state = trackToPredict.previousState;

                        % Predicting.
                        if (obj.SHOULD_PREDICT && any(trackToPredict.state ~= 0) && trackToPredict.predicted < obj.ALLOW_PRED_STEPS)
                            trackToPredict = obj.predictTrack(trackToPredict);
                            trackToPredict.predicted = trackToPredict.predicted + 1;
                        else
                            trackToPredict.isActive = 0;
                        end
                        
                        obj.setTrackById(trackToPredict.id, trackToPredict);
                    end
                else
                    % Stopping prediction.
                    ambTracks(1).predicted = 0;                    
                end   
                
                
            end
        end
        
        
        
        % Views the tracks on the frame.
        function viewActiveTracksOnFrame(obj, frameNumber, tracks)
            % First show the frame.
            %image(obj.getFilteredFrame(frameNumber)); truesize;
            imshow(obj.getRawFrame(frameNumber)); truesize;
            
            % Holding on for further annotations.
            hold on;
            % Title
            title(['Frame: ' num2str(frameNumber)]);
            
            activeTracks = tracks([tracks.isActive] == 1);
            
            % Then draw the tracks
            for i=1:length(activeTracks)
                
                pathX = activeTracks(i).path(activeTracks(i).path(:,2) ~= 0,2);
                pathY = activeTracks(i).path(activeTracks(i).path(:,2) ~= 0,3);
                
                % Drawing the path.
                plot(pathX,pathY,'r');
                % Drawing the last step.
                plot(activeTracks(i).lastCoordinates(1),activeTracks(i).lastCoordinates(2),'b+');
                % Plot the track ID
                text(activeTracks(i).lastCoordinates(1),activeTracks(i).lastCoordinates(2) - 30, ...
                    num2str(activeTracks(i).id),'color',[1 0 1]);
            end
            
            pause(0.01);
            
        end
        
        % Shows the original movie
        function showMovie(obj,pauseTime)
            if (nargin < 2)
                pauseTime = 0.001;
            end
            
            
            imgH = imshow(obj.getRawFrame(1));
            
            % Going over the frame range.
            for i=1:obj.numberOfFrames
                frame = obj.getRawFrame(i);
                
                set(imgH, 'CData', frame);
                title(['Frame: ' num2str(i)]);
                pause(pauseTime);
            end
        end
        
        % This function clears unesscary data in tracks
        function newTrack = cleanTrack(obj,oldTrack)
            newTrack = oldTrack;
            % Gets only valid path entries.
            newTrack.path = newTrack.path(newTrack.path(:,1) ~= 0,:);
            
            % Gets only valid step sizes entries.
            newTrack.stepSizes = newTrack.stepSizes(newTrack.stepSizes ~= 0);
            
            % Gets only the valid properties.
            newTrack.properties = newTrack.properties(newTrack.properties(:,1) ~= 0,:);
            
            % Setting "isUsed" property to clean
            newTrack.isUsed = 2;
        end
        
        function viewTracks3(obj,tracks,frameRange)
        end
        
        function [newFrame] = viewFrame3(obj,tracks,frameNumber)
            % Get the raw frame.
            rawFrame = obj.getRawFrame(frameNumber);
            
            if (size(rawFrame,3) == 1)
                rawFrame = repmat(rawFrame,[1 1 3]);
            end
            
            frameSize = size(rawFrame);
            
            % Look for relevant tracks.
            relevantTracksIds = arrayfun(@(t) any(t.path(:,1) == frameNumber), tracks);
            relevantTracks = tracks(relevantTracksIds);
            
            % Preparing the new frame.
            newFrame = rawFrame;
            
            for currentTrack=relevantTracks'
                pathToDraw = currentTrack.path(currentTrack.path(:,1)<=frameNumber,2:end);
                
                pathIndices = sub2ind(frameSize,round(pathToDraw(:,2)), round(pathToDraw(:,1)));
                
                redChannel = newFrame(:,:,1);
                greenChannel = newFrame(:,:,2);
                blueChannel = newFrame(:,:,3);
                redChannel(pathIndices) = 1;
                greenChannel(pathIndices) = 0;
                greenChannel(pathIndices) = 0;
                newFrame(:,:,1) = redChannel;
                newFrame(:,:,2) = greenChannel;
                newFrame(:,:,3) = blueChannel;
            end
            
            
            
            
        end
        
        
        % Views movie with given tracks
        function viewTracks(obj,tracks, frameRange, showCenterOfMass, videoSavePath, showLabels, hideTracks, scale)
            
            saveVideo = true;
            tracks = tracks(:);
            
            if (nargin < 7)
                hideTracks = false;
            end
            
            
            if (nargin < 6)
                showLabels = false;
            end
            
            if (nargin < 5)
                saveVideo = false;
            end
            
            if (nargin  < 4)
                showCenterOfMass = 0;
            end
            
            if (showLabels)
                [rightLabel,leftLabel] = obj.getLabels();
            end
            
            
            % Taking the least possible range.
            frameRange(2) = min(frameRange(2), obj.numberOfFrames);
            
            
            % Getting information about the size of the frame.
            if (isfield(obj.tracks(1),'parameters'))
                cords = obj.tracks(1).parameters.cords;
            else
                cords = [0 0; 0 0; 0 0];
            end
            
            if (isprop(obj,'parameters') &  isprop(obj.parameters,'Duration'))
                duration = obj.parameters.Duration;
            else
                % If no parameters so we set the duration = 1.
                duration = 1;
            end
            
            
            
            dimensions = size(obj.getRawFrame(1));
            cmPerPixel = 3 / pdist2(cords(1,:), cords(2,:));
            %pixelsPerCm = 1 / cmPerPixel;
            pixelsPerCm = 1;
            
            % Show image axis
            iptsetpref('ImshowAxesVisible','on');
            
            % Single Frame time ( as a fraction of a day ).
            singleFrameTime = (duration  / obj.numberOfFrames) / (24 * 60 * 60);
            
            % Clean tracks if nesscary.
            tracks = arrayfun(@(t) obj.cleanTrack(t),tracks);
            
            % Get tracks range.
            ranges = (arrayfun(@(x) [min(x.path(:,1)) max(x.path(:,1))], tracks,'UniformOutput',false));
            ranges = cell2mat(ranges);
            
            handles = [];
            
            if (saveVideo)
                if (hideTracks == true)
                    fileName = [fullfile(videoSavePath,obj.name) '.NoTracks.avi'];
                else
                    fileName = [fullfile(videoSavePath,obj.name) '.avi'];
                end
                vWriter = ...
                    VideoWriter(fileName, 'Motion JPEG AVI');
                
                %vWriter.FrameRate = (obj.numberOfFrames / duration) * 50;
                vWriter.FrameRate = 20;
                
                open(vWriter);
            end
            
            f = figure('units','normalized','outerposition',[0 0 1 1],'visible', 'on');
            
            imgH = imshow(obj.getRawFrame(frameRange(1)));
            
            % Going over the frame range.
            for i=frameRange(1):frameRange(2)
                % Logging
                disp(['Frame: ' num2str(i)]);
                
                frame = obj.getRawFrame(i);
                
                set(imgH, 'CData', frame);
                drawnow;
                hold on;
                
                
                % Get relevant tracks.
                relevantTracksIds = intersect(find(i >= ranges(:,1)), find(i<= ranges(:,2)));
                if (~isempty(handles))
                    arrayfun(@(x) delete(x), handles);
                end
                
                handles = [];
                if (hideTracks == false)
                    handles = zeros(length(relevantTracksIds) * 3,1);
                    for j=1:length(relevantTracksIds)                        
                        % Getting the current track.
                        currentTrack = tracks(relevantTracksIds(j));
                                               
                        % Getting the current position.
                        currentPos = find(currentTrack.path(:,1) == i);
                        
                        % Getting the path.
                        path = currentTrack.path(1:currentPos,:);
                        
                        % Drawing the path.
                        %if (currentTrack.id ~= 197)
                        handles((j-1) * 3 + 1) = plot(path(:,2),path(:,3),'r');
                        %else
                        %handles((j-1) * 3 + 1) = plot(path(:,2),path(:,3),'Color','Cyan','LineWidth',2);
                        %end
                        % Plotting the last step.
                        handles((j-1) * 3 + 2) = plot(path(currentPos,2),path(currentPos,3),'b+');
                        % Plot the track ID
                        handles(j * 3) = text(path(currentPos,2),path(currentPos,3) - 30, ...
                            num2str(currentTrack.id),'color',[1 0 1]);
                    end

                    
                    % If we want to show center of mass.
                    if (showCenterOfMass == 1)
                        currentPositions = ...
                            arrayfun(@(t) t.path(t.path(:,1) == i,2:3), tracks(relevantTracksIds),'UniformOutput',false);
                        
                        if (~isempty(currentPositions))
                            currentPositions = cell2mat(currentPositions);
                            
                            if (size(currentPositions,1) > 1)
                                centerPoint = mean(currentPositions);
                                
                                handles(length(handles) + 1) = plot(centerPoint(1),centerPoint(2),'g+','markersize',10);
                            end
                        end
                    end
                end
                
                curTimeStr = datestr(singleFrameTime * i, 'HH:MM:SS');
                %handles(length(handles) + 1) =...
                %text(50,50,[obj.name '  Frame: ' num2str(i)],'FontWeight','bold');
                %text(50,50,[obj.name '  Frame: ' num2str(i) ' Time: ' curTimeStr],'FontWeight','bold');
                
                
                % Writing the labels.
                if (showLabels)
                    handles(length(handles) + 1) = ...
                        text(cords(2,1) - 50, cords(2,2) + 220, rightLabel,'FontSize',15,'FontUnits','pixels');
                    
                    handles(length(handles) + 1) = ...
                        text(cords(3,1) - 50, cords(3,2) + 220, leftLabel,'FontSize',15,'FontUnits','pixels');
                end
                
                normalizedPixlPerCm = round(pixelsPerCm);
                
                if (normalizedPixlPerCm > 0)
                    roundDimensions = (dimensions) - mod(dimensions,normalizedPixlPerCm);
                    
                    xTicks = [1/5 * roundDimensions(2),...
                        2/5 * roundDimensions(2),...
                        3/5 * roundDimensions(2),...
                        4/5 * roundDimensions(2), ...
                        1 * roundDimensions(2)];
                    yTicks = [1/3 * roundDimensions(1),...
                        2/3 * roundDimensions(1),...
                        3/3 * roundDimensions(1)];
                    
                    % Once chan choose to put ticks accrdoing to the size
                    % of the field.
                    xTicks = [];
                    yTicks = [];
      
                             
                    set(gca,'XTick',xTicks);
                    set(gca,'XTickLabel',sprintf('%1d|',round(xTicks / normalizedPixlPerCm)));
                    set(gca,'YTick',yTicks);
                    set(gca,'YTickLabel',sprintf('%1d|',round(yTicks / normalizedPixlPerCm)));
                end
                
                set(gca,'FontSize',16);
                set(gca,'FontWeight','bold');
                
                %title(obj.name);
                %xlabel('X (pixels)');
                %ylabel('Y (pixels)');
                
                if (saveVideo)
                    frameImage = getframe(f);
                    
                    if (exist('scale','var'))
                        frameImage.cdata = imresize(frameImage.cdata,1/scale);
                    end
                    
                    writeVideo(vWriter, frameImage);
                end
                
                
                
                %Debug
                %pause;
                %Debug
            end
            
            if (saveVideo)
                close(vWriter);
            end
        end
        
        
        
        % This function creates a frame with a given tracks
        function frame = getTrackedFrame(obj, tracks, frameNumber)
            frame = ones(size(obj.videoRawData,1),size(obj.videoRawData,2),3) * 255;
            
            shapesToPrint = [];
            
            % See which entities we should draw on the frame.
            for i=1:length(tracks)
                curTrack = tracks{i};
                
                curEdge = curTrack(curTrack(:,1) == frameNumber,:);
                
                if (length(curEdge) == 0)
                    continue
                else
                    frameEntities = obj.allEntities{frameNumber};
                    shapesToPrint = [shapesToPrint;frameEntities{curEdge(3)}];
                end
            end
            
            frame(sub2ind(size(frame),shapesToPrint(:,2),shapesToPrint(:,1),shapesToPrint(:,3))) = 0;
            
            frame = frame / 255;
        end
        
        
        % The argument for this function should be monochrome image.
        function [centroids,areas, additionalProperties] = processEntitiesFromFrame(obj,frame, additional, smallestObj)
            % Just in case we get a
            if length(size(frame)) > 2
                frame = frame(:,:,1);
            end
            
            areas = [];
            
            % Copying the frame.
            binFrame = frame;
            additionalProperties = [];
            
            % Crating a binary image.
            binFrame(binFrame == 0) = 1;
            binFrame(binFrame == 255) = 0;
            
            if exist('smallestObj','var')
                binFrame = bwareaopen(binFrame, smallestObj);
            end
            
            % Now identifying the objects.
            %entities = bwconncomp(binFrame);
            
            if (additional == 2)
                % Create centroid out of the worms
                %preCentroids = regionprops(entities,'Centroid','Area','PixelList');
                %preCentroids = regionprops(entities,'Centroid','Area','MajorAxisLength', 'MinorAxisLength','PixelList','Solidity');
                preCentroids = regionprops(logical(binFrame),'Centroid','Area','MajorAxisLength', 'MinorAxisLength','PixelList','Solidity');
                
                % Saving other statistics
                additionalProperties = arrayfun(@(x)[{x.PixelList} {x.Area} {x.MajorAxisLength} {x.MinorAxisLength}], preCentroids, 'UniformOutput', false);
                
            elseif (additional == 1)
                % Create centroid out of the worms
                %preCentroids = regionprops(entities,'Centroid','Area','PixelList');
                %preCentroids = regionprops(entities,'Centroid','Area','MajorAxisLength', 'MinorAxisLength');
                preCentroids = regionprops(logical(binFrame),'Centroid','Area','MajorAxisLength', 'MinorAxisLength');
                
                % Saving other statistics
                additionalProperties = arrayfun(@(x)[x.Area x.MajorAxisLength x.MinorAxisLength], preCentroids, 'UniformOutput', false);
            elseif (additional == 3)
                %preCentroids = regionprops(entities,'Centroid', 'Area','MajorAxisLength','MinorAxisLength','Solidity','Eccentricity','EulerNumber');
                preCentroids = regionprops(logical(binFrame),'Centroid', 'Area','MajorAxisLength','MinorAxisLength','Solidity','Eccentricity','EulerNumber');
                
                additionalProperties = ...
                    arrayfun(@(record) [record.Area record.MajorAxisLength record.MinorAxisLength record.Solidity record.Eccentricity record.EulerNumber], ...
                    preCentroids,'UniformOutput', false);
                
            elseif (additional == 4)
                preCentroids = regionprops(logical(binFrame),'Centroid', 'Area','MajorAxisLength','MinorAxisLength','Solidity','Eccentricity');
                
                additionalProperties = ...
                    arrayfun(@(record) [record.Area record.MajorAxisLength record.MinorAxisLength record.Solidity record.Eccentricity], ...
                    preCentroids,'UniformOutput', false);
            end
            
            % Saving the information regarding the centroids (+ Area).
            centroids = arrayfun(@(x) [x.Centroid(1:2) x.Area],preCentroids, 'UniformOutput', false);
            centroids = cell2mat(centroids);
            
            if (~isempty(centroids))
                areas = centroids(:,3);
                centroids = centroids(:,1:2);
            end
            
        end
        
        
        % Collect worms statistics
        function [wormProperties] = chooseWorms(obj, frameNumber)
            % Getting the requested frame.
            rawFrame = obj.getRawFrame(frameNumber);
            filteredFrame = obj.getFilteredFrame(frameNumber);
            
            % Since we want to make sure we have RGB picture (as opposed to
            % monochromatic).
            frameSel(:,:,1:3) = repmat(filteredFrame(:,:,1),[1 1 3]);
            origFrameShow(:,:,1:3) = repmat(rawFrame(:,:,1), [1 1 3]);
            
            frameShow = origFrameShow;
            
            figure();
            %subplot(2, 2, [2 4]);
            imgShow = imshow(frameShow);
            title('Similiar Worms');
            
            %subplot(2, 2, [1 3]);
            figure();
            imgSel = imshow(frameSel);
            title('Choose typical worms');
            
            % Extracting properties.
            [~, ~, properties] = obj.processEntitiesFromFrame(frameSel,2);
            
            
            % We want to go until the user clicks the right button.
            button = 1;
            wormProperties = [];
            
            while (button ~= 3)
                
                [xUser,yUser,button] = ginput(1);
                
                % Looking at the convex Hull of the centroids
                %isInShape = cellfun(@(x) inpolygon(xUser,yUser,x{1}(:,1),x{1}(:,2)), properties);
                
                isInShape = cellfun(@(x) ismember([round(xUser),round(yUser)], x{1},'rows'), properties, 'UniformOutput', false);
                isInShape = cell2mat(isInShape);
                
                if (sum(isInShape ~= 0))
                    curShapeIndex = find(isInShape);
                    
                    shape = properties{curShapeIndex}{1};
                    
                    frameSel(sub2ind(size(frameSel),shape(:,2),shape(:,1),ones(size(shape,1),1) * 1)) = 255;
                    frameSel(sub2ind(size(frameSel),shape(:,2),shape(:,1),ones(size(shape,1),1) * 2)) = 0;
                    frameSel(sub2ind(size(frameSel),shape(:,2),shape(:,1),ones(size(shape,1),1) * 3)) = 0;
                    
                    %wormProperties = [wormProperties; [properties{curShapeIndex}{3},properties{curShapeIndex}{4}]];
                    wormProperties = [wormProperties; properties{curShapeIndex}{2:4}];
                    
                    % Updating the relevant worms.
                    %Now we'll calculate the distribution of the
                    shapesIndicesToDye = [];
                    
                    %Resetting the frame of the 'similiar worms'
                    frameShow = origFrameShow;
                    
                    if (size(wormProperties,1) == 1)
                        shapesIndicesToDye = curShapeIndex;
                    else
                        selShapesMeans = mean(wormProperties);
                        % Creating the covearince matrix (we adding a
                        % little bit to make it positive semi definite in
                        % case it isn't.
                        selShpaesCov = cov(wormProperties) + 0.001 * eye(size(wormProperties,2));
                        
                        
                        %shapesScore = cellfun(@(x) min(mvncdf([x{3:4}],selShapesMeans,selShpaesCov), 1-mvncdf([x{3:4}],selShapesMeans,selShpaesCov)), properties);
                        shapesScore = cellfun(@(x) min(mvncdf([x{2:4}],selShapesMeans,selShpaesCov), 1-mvncdf([x{2:4}],selShapesMeans,selShpaesCov)), properties);
                        
                        % Error threshold set to 0.05.
                        shapesIndicesToDye = find(shapesScore > 0.05);
                    end
                    
                    for t=1:length(shapesIndicesToDye)
                        shape = properties{shapesIndicesToDye(t)}{1};
                        
                        frameShow(sub2ind(size(frameSel),shape(:,2),shape(:,1),ones(size(shape,1),1) * 1)) = 255;
                        frameShow(sub2ind(size(frameShow),shape(:,2),shape(:,1),ones(size(shape,1),1) * 2)) = 0;
                        frameShow(sub2ind(size(frameShow),shape(:,2),shape(:,1),ones(size(shape,1),1) * 3)) = 0;
                    end
                    
                    
                    % Showing the selection frame
                    set(imgShow, 'CData', frameShow);
                    
                    % Showing similiar worms.
                    set(imgSel, 'CData', frameSel / 255);
                end
                
            end
            
        end
        
        function [wormProperties] = chooseWorms2(obj, frameNumber, relaySeq)
            if (exist('relaySeq','var') && relaySeq==false)
                binFrame = obj.getFilteredFrame(frameNumber, 0.98);
            else
                for i=max(1,frameNumber-6):frameNumber
                    binFrame = obj.getFilteredFrameSeq(i, 0.98);
                end
            end
            
            
            % Crating a binary image.
            binFrame(binFrame == 0) = 1;
            binFrame(binFrame == 255) = 0;
            
            binFrame = bwareaopen(binFrame,30);
            
            selectedObjects = bwselect(binFrame, 4);
            imshow(selectedObjects);
            %
            % Getting the entities.
            entities = bwconncomp(selectedObjects);
            
            
            preCentroids = regionprops(entities,'Area','MajorAxisLength','MinorAxisLength','Solidity','Eccentricity','EulerNumber');
            propCells = ...
                arrayfun(@(record) [record.Area record.MajorAxisLength record.MinorAxisLength record.Solidity record.Eccentricity record.EulerNumber], ...
                preCentroids,'UniformOutput', false);
            
            wormProperties = cell2mat(propCells);
        end
        
        
        function [wormProperties] = chooseWorms3(obj, frameNumber)
            for i=max(1,frameNumber-6):frameNumber
                binFrame = obj.getFilteredFrameSeq(i, 0.98);
            end
            
            % Crating a binary image.
            binFrame(binFrame == 0) = 1;
            binFrame(binFrame == 255) = 0;
            
            binFrame = bwareaopen(binFrame,90);
            
            [selectedObjects,idx] = bwselect(binFrame, 8);
            binFrame(idx) = 0;
            imshow(binFrame);
            
            
            % Getting the entities.
            entities = bwconncomp(binFrame);
            
            preCentroids = regionprops(entities,'Area','MajorAxisLength','MinorAxisLength','Solidity','Eccentricity','EulerNumber');
            propCells = ...
                arrayfun(@(record) [record.Area record.MajorAxisLength record.MinorAxisLength record.Solidity record.Eccentricity record.EulerNumber], ...
                preCentroids,'UniformOutput', false);
            
            wormProperties = cell2mat(propCells);
        end
        
        
        function [result] = mvncdfPar(obj, data, mean, covMat)
            result = zeros(1,size(data,1));
            
            parfor i=1:size(data,1)
                result(i) = mvncdf(data(i,:), mean,covMat);
            end
        end
        
        function [rightLabel, leftLabel] = getLabels(obj)
            % This is an helper function that returns the name of the
            % chemoattractant from the name of the obj.
            
            splits = strsplit(obj.name,'-');
            rightLabel = splits{6};
            commaPos = strfind(rightLabel,',');
            
            if (~isempty(commaPos))
                leftLabel = rightLabel(commaPos+1:end);
                rightLabel = rightLabel(1:commaPos-1);
            else
                if (length(splits) > 6)
                    leftLabel = splits{7};
                else
                    leftLabel = 'ETH';
                end
            end
            
            leftLabel = regexp(leftLabel,'[^ \.]*','match');
            rightLabel = regexp(rightLabel,'[^ \.]*','match');
            
            if length(rightLabel) > 1
                rightLabel = rightLabel{1};
            else
                rightLabel = cell2mat(rightLabel);
            end
            
            if length(leftLabel) > 1
                leftLabel = leftLabel{1};
            else
                leftLabel = cell2mat(leftLabel);
            end
            
            if (strfind(rightLabel,'DA')) rightLabel='DA'; end;
            if (strfind(rightLabel,'IA')) rightLabel='IAA'; end;
            if (strfind(rightLabel,'PD')) rightLabel='PD'; end;
            if (strfind(rightLabel,'MIX1')) rightLabel='DA + IAA'; end;
            
            if (strfind(leftLabel,'DA')) leftLabel='DA'; end;
            if (strfind(leftLabel,'IA')) leftLabel='IAA'; end;
            if (strfind(leftLabel,'PD')) leftLabel='PD'; end;
            if (strfind(leftLabel,'MIX1')) leftLabel='DA + IAA'; end;
        end
        
        
    end
    % This is being called when an object is being loaded from a file.s
    methods (Static)
      function obj = loadobj(s)
        s.initialize();
      end
   end
end

