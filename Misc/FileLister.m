classdef FileLister
    %FILELISTER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        rootDirectory
        fileDesciptor
        
        allFiles
    end
    
    methods
        function FL = FileLister(rootDirectory,fileDesciptor)
            FL.rootDirectory = rootDirectory;
            FL.fileDesciptor = fileDesciptor;
            
            FL.allFiles = FL.listFiles(rootDirectory,fileDesciptor);
        end
        
        function [files] = listFiles(obj, rootDirectory, filesDescriptor)            
            % Local Files
            files = dir(fullfile(rootDirectory, filesDescriptor));
            
            for i=1:length(files)
                files(i).name = fullfile(rootDirectory,files(i).name);
            end
            
            
            % Local Dirs 
            rootEntities = dir(rootDirectory);
            dirs = rootEntities([rootEntities.isdir]);
            dirs = dirs(3:end);
            
            % Iterating over local dirs
            for d=dirs'
                files = [files; obj.listFiles(fullfile(rootDirectory, d.name), filesDescriptor)];
            end
        end
        
        
    end
    
end

