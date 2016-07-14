function [ xPos, yPos,xVec,yVec, background] = attractionVectorField( tracker, flip )
%SHOWQUIVER Summary of this function goes here

if (~exist('flip','var'))
    flip = 0;
end


f = figure('units','normalized','outerposition',[0 0 1 1],'visible', 'on');
    
    tracks = tracker.tracks;
    
    frameSize = size(tracker.getRawFrame(1));
    
    blockSizeX = 75;
    blockSizeY = 76;
    
    xBoxes = floor(frameSize(2) / blockSizeX) + 1;
    yBoxes = floor(frameSize(1) / blockSizeY) + 1;
    
    quivers = cell(yBoxes,xBoxes);
    
    for currentTrack=tracks'
        steps = diff(currentTrack.path(:,3:-1:2));
        
        if (size(currentTrack.path,1) == 0)
            continue;
        end
        
        
        for i=1:(size(currentTrack.path,1) - 1)
            pos = currentTrack.path(i,2:3);
            
            cellPosX = max(floor(pos(1) / (frameSize(2) / (xBoxes - 1))) + 1,1);
            cellPosY = max(floor(pos(2) / (frameSize(1) / (yBoxes - 1))) + 1,1);
            
            if (cellPosY > yBoxes || cellPosX > xBoxes)
                continue;
            end
                
            
            if (flip == 1)
                steps(i,2) = -steps(i,2);
            end
            quivers{cellPosY, cellPosX} = [quivers{cellPosY, cellPosX}; steps(i,:)];
        end
        
    end
    
    boxCount = cellfun(@(c) size(c,1), quivers);
    boxCount = min(boxCount, quantile(boxCount(:), 0.85));
    quivers = cellfun(@(c) mean(c,1), quivers,'UniformOutput', false);
    quivers = cellfun(@(c) (c/norm(c)) * 0.5, quivers,'UniformOutput', false);
    
    
    [yPos,xPos] = meshgrid(1:yBoxes,1:xBoxes);
    xPos = xPos';
    yPos = yPos';
    
    xPos = xPos(:);
    yPos = yPos(:);
    
    
    xVec =[];
    yVec = [];
    for j=1:xBoxes
        for i=1:yBoxes
            if (~isnan(quivers{i,j}))
                xVec = [xVec quivers{i,j}(2)];
                yVec = [yVec quivers{i,j}(1)];
            else
                xVec = [xVec 0];
                yVec = [yVec 0];
            end
        end
    end
    
    if (flip == 1)
        xPos = (xBoxes - xPos + 1);
        boxCount =  fliplr(boxCount);
    end

    
    colormap(winter);
    imagesc(boxCount); hold on;
    colorbar;
    qp = quiver(xPos, yPos,xVec',yVec',0.4,'w', 'MaxHeadSize',0.05, 'LineWidth', 0.75, 'MarkerSize',200);
    set(qp,'linewidth',2);
    xlim([1,xBoxes]);
    ylim([1,yBoxes]);
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    title('Binned Averaged Direction');
    
    background = boxCount;
    %save('attractionField.mat','xPos','yPos','xVec','yVec','background');
end


