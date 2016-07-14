function [ wormsAtRegion ] = polygonDynamics( tracker, shouldPlot )
%POLYGONDYNAMICS
% This method returns the number of worms in a specific polygon in the
% field over time.

% Showing the first frame.
h = figure();
imshow(tracker.getRawFrame(1));
title('Please mark a polygon (Right click on the image when finished');

% Marking polygone.
[polyX, polyY] = getline('closed');
close(h);

% The range of frames we'll iterate over.
rangeOfFrames = 1:tracker.numberOfFrames;

% Relevant tracks.
refTracks = tracker.tracks;

% Initializing
wormsAtRegion = zeros(1,length(rangeOfFrames));

% Tracks in and out
tracksInOut = cell(2,length(rangeOfFrames));

% Tracks in frames
tracksInFrames = zeros(length(refTracks), length(rangeOfFrames));

% Tracks positions
tracksEntries = zeros(length(refTracks),length(rangeOfFrames));

for i=1:length(refTracks)
    curTrack = refTracks(i);
    frames = curTrack.path(:,1);
    
    tracksInFrames(i, frames) = 1;
    
    tracksEntries(i,curTrack.path(1,1):curTrack.path(end,1)) = inpolygon(curTrack.path(:,2), curTrack.path(:,3),polyX, polyY);
end


% Running over the frame
for i=2:length(rangeOfFrames)
    
    
    relevantTracksIds = find(tracksInFrames(:,i));
    
    nearPole = relevantTracksIds(logical(tracksEntries(relevantTracksIds,i)));
    farPole = relevantTracksIds(logical(~tracksEntries(relevantTracksIds,i)));
    
    
    % Checking for entrances and exists.
    enterances = intersect(tracksInOut{2}, nearPole);
    exists = intersect(tracksInOut{1}, farPole);
    
    wormsAtRegion(i) = wormsAtRegion(i - 1) + length(enterances) - length(exists);
    
    tracksInOut{1} = nearPole;
    tracksInOut{2} = farPole;
    
    
    disp(['Frame number: ' num2str(i) ' Region:' num2str(wormsAtRegion(i)')]);
end

if ( nargin > 1 && shouldPlot ~= 0 )
   plot(wormsAtRegion,'LineWidth',2);
   xlabel('Frame','FontSize',12);
   ylabel('Worms in polygon','FontSize',12);
   xlim([1, tracker.numberOfFrames]);
end


end

