classdef AnimalsRecorder < handle
    %WORMSRECORDER Summary of this class goes here
    %   Detailed explanation goes here
    properties
        sTime
        
        currentSessionName
        
        inputVideo
        
        moviesDir
        
        fixedParameters 
        
        micName
        
        currentFile
        
        % Here we store parameters we want to add
        % during the acquisition of the video.
        onlineParameters
    end
    methods
        function WR = AnimalsRecorder(micName, moviesDir, exposure, fixedParameters)
            WR.inputVideo = videoinput('qimaging', 1, 'MONO8_2560x1920');
            src = getselectedsource(WR.inputVideo);
            src.Exposure = exposure;

            WR.inputVideo.FramesPerTrigger = Inf;
            WR.inputVideo.LoggingMode = 'disk';
            WR.inputVideo.FrameGrabInterval = 5;
            WR.micName = micName;
            %WR.inputVideo.FrameGrabInterval = 5;
            
            WR.moviesDir = moviesDir;
            
            WR.onlineParameters = {};
            
            if nargin < 4
                fixedParameters.chemoV = 10;
            end
            
            fixedParameters.experimentDate = date;
            % This line is relevant for worm breeders.
            %fixedParameters.bleachTime = datevec(guiDatePicker('1-Jan-2014'));
            
            WR.fixedParameters = fixedParameters;
        end
        
        function close(obj)
            delete(obj.inputVideo);
        end
        
        function preview(obj)
            preview(obj.inputVideo);
        end
        
        
        function startVideo(obj,name)
            % Saving video positions.
            obj.addPositions()
            
            obj.currentSessionName = name;
                                    
            % DateStr
            dateStr = datestr(clock);
            dateStr = strrep(dateStr,' ','-');
            dateStr = strrep(dateStr,':','.');
            
            % the complete movie file name.
            obj.currentFile = [dateStr '-' obj.micName '-' name];
            fileName = fullfile(obj.moviesDir, [obj.currentFile '.avi']);
            
            % Disk logging
            diskLogger = VideoWriter(fileName, 'Archival');
            diskLogger.FrameRate = 10;
            %diskLogger.FrameRate = 1;
            obj.inputVideo.DiskLogger = diskLogger;
            obj.preview();
            
            obj.sTime = clock();
            start(obj.inputVideo);
            
            %hoursSinceBleach = etime(clock, obj.fixedParameters.bleachTime) / 60 / 60;
            %obj.addParameter('HoursSinceBleach', hoursSinceBleach);
        end
        
        function endVideo(obj)
            % Taking the time.
            eTime = clock();
            
            % Stoping the acquisition.
            stop(obj.inputVideo);
            
            % Parameters file name.
            parametersFileName = fullfile(obj.moviesDir, [obj.currentFile '.mat']);

            duration = etime(eTime,obj.sTime);
            
            parameters = obj.fixedParameters;
            parameters.StartTime = obj.sTime;
            parameters.EndTime = eTime;
            parameters.Duration = duration;
            
            for i=1:length(obj.onlineParameters)
                parameters.(obj.onlineParameters{i}{1}) = obj.onlineParameters{i}{2};
            end
            
            save(parametersFileName,'parameters');
        end
        
        function addParameter(obj,key,value)
            obj.onlineParameters{length(obj.onlineParameters) + 1}...
                = {key, value};
        end
        
        
        function addPositions(obj)
            firstFrame =  getsnapshot(obj.inputVideo);
            
            fig = figure();
            imshow(imresize(firstFrame, 0.5));
            title('Choose points');
            obj.addParameter('cords', ginput(3) * 2);
            close(fig);
        end
    end
    
end

