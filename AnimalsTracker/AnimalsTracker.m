function varargout = AnimalsTracker(varargin)
%ANIMALSTRACKER M-file for AnimalsTracker.fig
%      ANIMALSTRACKER, by itself, c
% a new ANIMALSTRACKER or raises the existing
%      singleton*.
%
%      H = ANIMALSTRACKER returns the handle to a new ANIMALSTRACKER or the handle to
%      the existing singleton*.
%
%      ANIMALSTRACKER('Property','Value',...) creates a new ANIMALSTRACKER using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to AnimalsTrafrcker_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      ANIMALSTRACKER('CALLBACK') and ANIMALSTRACKER('CALLBACK',hObject,...) call the
%      local function named  CALLBACK in ANIMALSTRACKER.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help AnimalsTracker

% Last Modified by GUIDE v2.5 15-Jul-2015 13:16:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @AnimalsTracker_OpeningFcn, ...
                   'gui_OutputFcn',  @AnimalsTracker_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before AnimalsTracker is made visible.
function AnimalsTracker_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for AnimalsTracker
handles.output = hObject;

% Storing handles to the image figures.
axes(handles.filteredFrame);
handles.filteredImageHandle = 0;
axes(handles.rawFrame);
handles.rawImageHandle = 0;
handles.filteredImageHandle = 0;
handles.zoomedImageHandle = 0;

% Update handles structure
guidata(hObject, handles);


% Remove the ticks from the axes.
set(handles.filteredFrame,'xtick',[])
set(handles.filteredFrame,'xticklabel',[])
set(handles.filteredFrame,'ytick',[])
set(handles.filteredFrame,'yticklabel',[])

% Remove the ticks from the axes.
set(handles.rawFrame,'xtick',[])
set(handles.rawFrame,'xticklabel',[])
set(handles.rawFrame,'ytick',[])
set(handles.rawFrame,'yticklabel',[])

% Remove the ticks from the axes.
set(handles.zoomedFrame,'xtick',[])
set(handles.zoomedFrame,'xticklabel',[])
set(handles.zoomedFrame,'ytick',[])
set(handles.zoomedFrame,'yticklabel',[])


set (gcf, 'WindowButtonMotionFcn', @filteredFrameMouseMove);
set(handles.logBox,'string',[{'Starting..'}]);

set(handles.showTracksButton,'Enable','off');

% UIWAIT makes AnimalsTracker wait for user response (see UIRESUME)
% uiwait(handles.figure1);



% --- Outputs from this function are returned to the command line.
function varargout = AnimalsTracker_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;





% --- Executes during object creation, after setting all properties.
function stdSlider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stdSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% % --- Executes on slider movement.
% function thrSlider_Callback(hObject, eventdata, handles)
% % hObject    handle to thrSlider (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % Hints: get(hObject,'Value') returns position of slider
% %        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
% value = get(handles.thrSlider,'Value');
% handles.currentFilterParameters.threshold = value;
% handles.currentFilterParameters.userModifiedThr = true;
% handles.newFrame = false;
% thrSliderChangeValue(hObject,value, true);
% guidata(hObject,handles);

% 
% function thrSliderChangeValue(hObject,value, showFilteredFrame)
% 
% handles = guidata(hObject);
% 
% set(handles.thrText,'String', num2str(value));
% set(handles.thrSlider,'Value',value);
% handles.currentFilterParameters.threshold = value;
% guidata(hObject,handles)
% 
% if (showFilteredFrame)
%     % Updating the filtered frame.
%     updateFilteredFrame(hObject);
% end



% --- Executes on slider movement.
function stdSlider_Callback(hObject, eventdata, handles)
% hObject    handle to stdSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
value = get(handles.stdSlider,'Value');
guidata(hObject, handles);

stdSliderChangeValue(hObject,value,true);



function stdSliderChangeValue(hObject, value, showFilteredFrame)
handles = guidata(hObject);

set(handles.stdText,'String', num2str(value));
set(handles.stdSlider,'Value', value);
handles.currentFilterParameters.logStd = value;
guidata(hObject,handles)

if (showFilteredFrame)
    % Updating the filtered frame.
    updateFilteredFrame(hObject);
    % thrSliderChangeValue(hObject, thr, false);
end




% --- Executes on slider movement.
function frameSlider_Callback(hObject, eventdata, handles)
% hObject    handle to frameSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATsA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
value = round(get(handles.frameSlider,'Value'));

frameSliderChangeValue(hObject,value);




function frameSliderChangeValue(hObject,value)
handles = guidata(hObject);

% Disabling the threshold slider and edit text.
% set(handles.thrSlider,'Enable','off');
% set(handles.thrText,'Enable','off');
% set(handles.manualThresholdingCheck,'Value',0);

set(handles.framesText,'String', num2str(value));
set(handles.frameSlider,'Value', value);

handles.currentFrame = value;
guidata(hObject, handles);

% Changing the raw frame.
updateRawFrame(hObject);

handles = guidata(hObject);

% These are the default values for unset frames.
stdValue = 3.55;

% Checking to see if there are any marks for this frame
if (~isempty(handles.framesMarks(handles.currentFrame).marksIndices))
    set(handles.unmarkWormButton,'Enable','on');
else
    set(handles.unmarkWormButton,'Enable','off');
end

% Defaultly we're not showing the show identity worm button as active.
set(handles.showIdenButton,'Enable','off');

% Checking to see if there are any saved values.
if (handles.framesFilterParameters(handles.currentFrame).logStd ~= 0)
    if (~isempty(handles.allFeatures))
        set(handles.showIdenButton,'Enable','on');
    else
        set(handles.showIdenButton,'Enable','off');
    end

    
    handles.currentFilterParameters = handles.framesFilterParameters(handles.currentFrame);

    % Enabling the user to mark worms.
    set(handles.markWormsButton,'Enable','on');
    guidata(hObject, handles);
else
    handles.currentFilterParameters.logStd = stdValue;
    
    % Disenabling the user to mark worms.
    set(handles.markWormsButton,'Enable','off');

    guidata(hObject, handles);
end

updateFilteredFrame(hObject);


stdSliderChangeValue(hObject,handles.currentFilterParameters.logStd, false);



% --- Executes during object creation, after setting all properties.
function frameSlider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to frameSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end





% --------------------------------------------------------------------
function fileMenu_Callback(hObject, eventdata, handles)
% hObject    handle to fileMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
% The open menu option.
% --------------------------------------------------------------------
function openMenu_Callback(hObject, eventdata, handles)
% hObject    handle to openMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get the supported file formats.
formats = VideoReader.getFileFormats();
% Convert to a filter list.
filterSpec = getFilterSpec(formats);

[file, folder] = uigetfile(filterSpec, 'Open a movie file');

if (file ~= 0)
    fullFileName = fullfile(folder, file);
    
    log(handles,'Starting to load video file..');
    try
        % Accessing the file using a videoReader
        videoTracker = VideoTracker(fullFileName);
    catch err
        msgbox(['Cannot open a video file: ' err.message]);
        return;
    end
    log(handles,'Loading completed.');
    
    
    % Storing the full file name.
    handles.fullFileName = fullFileName;
    % Storing the video reader.
    handles.videoTracker = videoTracker;
    % Saves the videos frame size.
    handles.frameSize = size(videoTracker.getRawFrame(1));
    % Saving the initial filter params structure.
    emptyParamsStruct = struct('logStd',0);
    handles.framesFilterParameters = repmat(emptyParamsStruct, videoTracker.numberOfFrames, 1);
    handles.currentFilterParameters = emptyParamsStruct;
    %Saving the initial marks structure
    emptyMarksStruct = struct('marksIndices',[],'marksFeatures', []);
    handles.framesMarks = repmat(emptyMarksStruct, videoTracker.numberOfFrames, 1);
    % CurrentFrame number
    handles.currentFrame = 1;
    % Marked features.
    handles.allFeatures = [];
    
    
    % Saving guidata.
    guidata(hObject,handles);
    
    % Initialize the initial state of the GUI.
    initializeUserInterface(hObject, handles);
    
    % Opening the first frame
    frameSliderChangeValue(hObject,1);
end


% --------------------------------------------------------------------
% Updating the raw frame image.
% --------------------------------------------------------------------
function updateRawFrame(hObject)
   

handles = guidata(hObject);

% Getting the current raw frame.
currentFrameData = handles.videoTracker.getRawFrame(handles.currentFrame);

% Read the current frame.
if (handles.rawImageHandle == 0)
    axes(handles.rawFrame);
    imgH = imshow(currentFrameData);
    handles.rawImageHandle = imgH;
    guidata(hObject,handles);
    
    set(handles.rawFrame,'xtick',[])
    set(handles.rawFrame,'xticklabel',[])
    set(handles.rawFrame,'ytick',[])
    set(handles.rawFrame,'yticklabel',[])
else
    set(handles.rawImageHandle,'CData',currentFrameData);
    drawnow;
end

function [thr] = updateFilteredFrame(hObject, markIdxs)
handles = guidata(hObject);

currentFrame = handles.currentFrame;


[filteredFrameData,thr] = handles.videoTracker.getFilteredFrame2(currentFrame, handles.currentFilterParameters.logStd);

% Create an RGB frame.
rgbFrame = repmat(filteredFrameData,[1 1 3]);
% Marking in green.
redFrame = rgbFrame(:,:,1);
greenFrame = rgbFrame(:,:,2);
blueFrame = rgbFrame(:,:,3);


% Checking if we marked shapes on this frames.
frameIndices = handles.framesMarks(handles.currentFrame).marksIndices;
if (~isempty(frameIndices))
    
    for i=1:length(frameIndices)
        currentIndices = frameIndices{i};
        greenFrame(currentIndices) = 1;
        %redFrame(currentIndices) = 1;
        %blueFrame(currentIndices) = 0;
    end
    
end

% Marking special marks.
if (exist('markIdxs','var'))
    for i=1:length(markIdxs)
        %redFrame(markIdxs{i}) = 1;
        %greenFrame(markIdxs{i}) = 0;
        blueFrame(markIdxs{i}) = 1;
    end
end


% Returning the channels into one frame.
rgbFrame(:,:,1) = redFrame;
rgbFrame(:,:,2) = greenFrame;
rgbFrame(:,:,3) = blueFrame;
filteredFrameData = double(rgbFrame);




if (handles.filteredImageHandle == 0)
    axes(handles.filteredFrame);
    imgH = imshow(filteredFrameData);
    handles.filteredImageHandle = imgH;
    guidata(hObject,handles);
    set(handles.filteredFrame,'xtick',[]);
    set(handles.filteredFrame,'xticklabel',[]);
    set(handles.filteredFrame,'ytick',[]);
    set(handles.filteredFrame,'yticklabel',[]);
else
    set(handles.filteredImageHandle,'CData',filteredFrameData);
end










% --------------------------------------------------------------------
% Initialize the user interface
% --------------------------------------------------------------------
function initializeUserInterface(hObject, handles)
% handles    structure with handles and user data (see GUIDATA)

% Initializign the frames slider.
numberOfFrames = handles.videoTracker.numberOfFrames;

set(handles.frameSlider, 'Value', 1);
set(handles.frameSlider, 'Max',numberOfFrames);
set(handles.frameSlider, 'Min',1);
set(handles.frameSlider, 'SliderStep', [1/numberOfFrames,  10/numberOfFrames]);

% Initializing the threshold sliders.
% Gaussian std slider.
minStd = 1;
maxStd = 30;
startStd = 4;
stdResolution = 1000;
set(handles.stdSlider, 'Value', startStd);
set(handles.stdSlider, 'Max',maxStd);
set(handles.stdSlider, 'Min',minStd);
set(handles.stdSlider, 'SliderStep', [1 / stdResolution , 10 / stdResolution]);
% Binary threshold slider.
% thresholdResolution = 10000;
% set(handles.thrSlider, 'Value', 0.5);
% set(handles.thrSlider, 'Max',1);
% set(handles.thrSlider, 'Min',0);
% set(handles.thrSlider, 'SliderStep', [1/thresholdResolution , 10/thresholdResolution]);

guidata(hObject, handles);


function filteredFrameMouseMove(hObject, eventdata, handles)

handles = guidata(hObject);

if (handles.filteredImageHandle == 0)
    return;
end

% Getting the pxiel position of the filtered frame object
oldunits = get(handles.filteredFrame, 'Units');
set(handles.filteredFrame, 'Units', 'pixels');
axesLocation = get(handles.filteredFrame, 'Position');
set(handles.filteredFrame, 'Units', oldunits);

mousePosition = get (handles.filteredFrame, 'CurrentPoint');
mousePosition = mousePosition(1,:);
%disp(num2str(mousePosition));

if (all(mousePosition(1:2) > 0) && all(mousePosition(1:2) <= fliplr(handles.frameSize)))
    %disp('1');
    updateZoomFrame(hObject,round(mousePosition));
end

function [X,Y keydown] = altGInput(obj, limits)
prevPointer = get(gcf,'Pointer')
set(gcf,'Pointer','crosshair')
keydown = waitforbuttonpress;
button = get(gcf, 'SelectionType');
mousePosition = get(obj, 'CurrentPoint');
X = mousePosition(1,1);
Y = mousePosition(1,2);

if (strcmp(button,'normal'))
    keydown = 1;
elseif (strcmp(button,'alt'))
    keydown=2;
end

set(gcf,'Pointer',prevPointer)

% if (mousePosition(2) > axesLocation(1) && mousePosition(2) < (axesLocation(1) + axesLocation(3)) ...
%         && mousePosition(1) > axesLocation(2) && mousePosition(1) < (axesLocation(2) + axesLocation(4)))
%     disp('1');
% else
%     disp('0');
% end



% --------------------------------------------------------------------
% Initialize the user interface
% --------------------------------------------------------------------
function updateZoomFrame(hObject, pos)
MAGNIFICATION_FACTOR = 3;

handles = guidata(hObject);

currentFrame = handles.currentFrame;

% Magnified pos
newPos = MAGNIFICATION_FACTOR * pos;

% Magnifing the frame.
if (~isfield(handles,'magnifiedFrameNumber') || (size(handles.magnifiedFrameNumber,1) == 0) || ...
        (handles.magnifiedFrameNumber ~= currentFrame))
    [filteredFrameData,~] = handles.videoTracker.getFilteredFrame2(currentFrame, handles.currentFilterParameters.logStd);
    magnifiedFrame = imresize(filteredFrameData,MAGNIFICATION_FACTOR);
    handles.magnifiedFrame = magnifiedFrame;
    handles.magnifiedFrameNumber = currentFrame;
else
    magnifiedFrame = handles.magnifiedFrame;
end

rangeX = max((newPos(2)-100):(newPos(2)+100),1);
rangeX = min(rangeX,handles.frameSize(1) * MAGNIFICATION_FACTOR);
rangeY = max((newPos(1)-100):(newPos(1)+100),1);
rangeY = min(rangeY,handles.frameSize(2) * MAGNIFICATION_FACTOR);

smallFrame = magnifiedFrame(rangeX,rangeY);

if (handles.zoomedImageHandle == 0)
    axes(handles.zoomedFrame);
    imgH = imshow(smallFrame);
    handles.zoomImageHandle = imgH;
    guidata(hObject,handles);
    
    set(handles.rawFrame,'xtick',[])
    set(handles.rawFrame,'xticklabel',[])
    set(handles.rawFrame,'ytick',[])
    set(handles.rawFrame,'yticklabel',[])
else
    set(handles.zoomImageHandle,'CData',currentFrameData);
end

line([100,100],[0,200],'color','red');
line([0,200],[100,100],'color','red');

guidata(hObject,handles);









% --- Executes during object creation, after setting all properties.
function framesText_CreateFcn(hObject, eventdata, handles)




function thrText_Callback(hObject, eventdata, handles)
% hObject    handle to thrText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of thrText as text
%        str2double(get(hObject,'String')) returns contents of thrText as a double
try
    inputValue = str2double(get(handles.thrText,'String'));
catch 
    msgbox('Invalid value.')
end

% We have to check if this value is in range
minLegal = get(handles.thrSlider,'Min');
maxLegal = get(handles.thrSlider,'Max');

if ((inputValue<minLegal | inputValue >maxLegal) | isnan(inputValue))
    msgbox('Invalid value.')
else
    thrSliderChangeValue(hObject,inputValue, true);
end


function framesText_Callback(hObject, eventdata, handles)
try
    inputValue = ceil(str2double(get(handles.framesText,'String')));
catch 
    msgbox('Invalid value.')
end

% We have to check if this value is in range
minLegal = get(handles.frameSlider,'Min');
maxLegal = get(handles.frameSlider,'Max');

if ((inputValue<minLegal | inputValue >maxLegal) | isnan(inputValue))
    msgbox('Invalid value.')
else
    frameSliderChangeValue(hObject,inputValue);
end

function stdText_Callback(hObject, eventdata, handles)
% hObject    handle to thrText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of thrText as text
%        str2double(get(hObject,'String')) returns contents of thrText as a double
try
    inputValue = str2double(get(handles.stdText,'String'));
catch 
    msgbox('Invalid value.')
end

% We have to check if this value is in range
minLegal = get(handles.stdSlider,'Min');
maxLegal = get(handles.stdSlider,'Max');

if ((inputValue<minLegal | inputValue >maxLegal) | isnan(inputValue))
    msgbox('Invalid value.')
else
    stdSliderChangeValue(hObject,inputValue, true);
end

    



% --- Executes during object creation, after setting all properties.
function thrText_CreateFcn(hObject, eventdata, handles)
% hObject    handle to thrText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end








% --- Executes during object creation, after setting all properties.
function stdText_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stdText (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in manualThresholdingCheck.
function manualThresholdingCheck_Callback(hObject, eventdata, handles)
% hObject    handle to manualThresholdingCheck (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of manualThresholdingCheck
thresholding = get(handles.manualThresholdingCheck,'Value');

if (thresholding == 1)
    set(handles.thrSlider,'Enable','on')
    set(handles.thrText,'Enable','on')
else
    set(handles.thrSlider,'Enable','off')
    set(handles.thrText,'Enable','off')
end



% --- Executes on button press in thrSaveButton.
function thrSaveButton_Callback(hObject, eventdata, handles)
% hObject    handle to thrSaveButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.framesFilterParameters(handles.currentFrame) = handles.currentFilterParameters;
set(handles.markWormsButton,'Enable','on');
guidata(hObject,handles);

if (~isempty(handles.allFeatures))
    set(handles.showIdenButton,'Enable','on');
else
    set(handles.showIdenButton,'Enable','off');
end


% --- Executes on button press in saveForNextFrames.
function saveForNextFrames_Callback(hObject, eventdata, handles)
% hObject    handle to saveForNextFrames (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
nextFrames = handles.currentFrame:handles.videoTracker.numberOfFrames;
handles.framesFilterParameters(nextFrames) = repmat(handles.currentFilterParameters, length(nextFrames),1);
set(handles.markWormsButton,'Enable','on');
% Forcing the magnigied frame to update.
handles.magnifiedFrameNumber = [];

% Updating handles.
guidata(hObject, handles);

if (~isempty(handles.allFeatures))
    set(handles.showIdenButton,'Enable','on');
else
    set(handles.showIdenButton,'Enable','off');
end


% --- Executes on button press in unmarkWormButton.
function unmarkWormButton_Callback(hObject, eventdata, handles)
% hObject    handle to unmarkWormButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.framesMarks(handles.currentFrame).marksIndices = [];
handles.framesMarks(handles.currentFrame).marksFeatures = [];

% We consolidate all the statistics.
cellAllFeatures = arrayfun(@(f) f.marksFeatures, handles.framesMarks,'UniformOutput',false);
handles.allFeatures = cell2mat(cellAllFeatures);
handles.videoTracker.learn2(handles.allFeatures);
guidata(hObject, handles);

if (isempty(handles.allFeatures))
    set(handles.showIdenButton,'Enable','off');
end

log(handles, ['Cleaned worm marks from frame: ' num2str(handles.currentFrame)]);
updateFilteredFrame(hObject);


% --- Executes on button press in markWormsButton.
function markWormsButton_Callback(hObject, eventdata, handles)
% hObject    handle to markWormsButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.filteredFrame);
log(handles,['Marking worms on frame: ' num2str(handles.currentFrame)]);
log(handles,['(Press the right mouse key to quit marking mode)']);

% Getting the saved gaussian std for this frame.
gaussianStd = handles.framesFilterParameters(handles.currentFrame).logStd;
button = 1;
[x,y,button] = altGInput(handles.filteredFrame);

% Gettign the frames features.
[~,features,extermas,idxs] = handles.videoTracker.extractFeatures2(handles.currentFrame, gaussianStd);

% We're removing some quantile of the biggest areas in the video.
AREA_QUANTILE_REMOVE = 0.999;
areas = cellfun(@(exterma) polyarea(exterma(:,1),exterma(:,2)), extermas);
thresholdArea = quantile(areas,AREA_QUANTILE_REMOVE);

features(areas > thresholdArea,:) = [];
extermas(areas > thresholdArea,:) = [];

idxs(areas > thresholdArea,:) = [];



% Getting the frame
frame = handles.videoTracker.getFilteredFrame2(handles.currentFrame, gaussianStd);

rgbFrame = repmat(frame,[1 1 3]);

while (button == 1)
    % Getting the rounded coordiantes.
    x = round(x);
    y = round(y);
        
    % Looking for figure that matches that choice.
    matchingRegions = cellfun(@(exterma) inpolygon(x,y,exterma(:,1), exterma(:,2)), extermas);
    
    % If we're inside more than one polygon.
    if (any(matchingRegions))
        ids = find(matchingRegions);
        areas = arrayfun(@(id) polyarea(extermas{id}(:,1),extermas{id}(:,2)), ids);
        [~,i] = min(areas);
        matchingRegions = logical(zeros(size(matchingRegions)));
        matchingRegions(ids(i)) = true;
    end
    
    
    % Checking both that we have a region and that we hit an actual marked
    % region.
    if (any(matchingRegions) && (frame(y,x) == 0))
        handles.framesMarks(handles.currentFrame).marksIndices = [handles.framesMarks(handles.currentFrame).marksIndices idxs(matchingRegions)];
        handles.framesMarks(handles.currentFrame).marksFeatures = [handles.framesMarks(handles.currentFrame).marksFeatures;features(matchingRegions,:)];
        
        redFrame = rgbFrame(:,:,1);
        greenFrame = rgbFrame(:,:,2);
        blueFrame = rgbFrame(:,:,3);
        
        currentIdxs = idxs{matchingRegions};
        greenFrame(currentIdxs) = 1;
        blueFrame(currentIdxs) = 0;
        redFrame(currentIdxs) = 0;
        
        rgbFrame(:,:,1) = redFrame;
        rgbFrame(:,:,2) = greenFrame;
        rgbFrame(:,:,3) = blueFrame;
        
        set(handles.filteredImageHandle,'CData',rgbFrame);
        % The first features is always the area.
        log(handles,['Marked worm. Worms'' area: ' num2str(features(matchingRegions,1)) ' pixels.' ]);
    end
    
    
    [x,y,button] = altGInput(handles.filteredFrame);
end

% We consolidate all the statistics.
cellAllFeatures = arrayfun(@(f) f.marksFeatures, handles.framesMarks,'UniformOutput',false);
handles.allFeatures = cell2mat(cellAllFeatures);
handles.videoTracker.learn2(handles.allFeatures);
guidata(hObject, handles);

log(handles,[num2str(size(handles.allFeatures,1)) ' worms were marked.']);

updateFilteredFrame(hObject);

% Enabling the show animals button.
if (~isempty(handles.allFeatures))
    set(handles.showIdenButton,'Enable','on');
else
    set(handles.showIdenButton,'Enable','off');
end


% Enabling the unmark
if (~isempty(handles.framesMarks(handles.currentFrame).marksIndices))
    set(handles.unmarkWormButton,'Enable','on');
end



% --- Executes on button press in showIdenButton.
function showIdenButton_Callback(hObject, eventdata, handles)
% hObject    handle to showIdenButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Temporary implementation.

% Getting the saved gaussian std for this frame.
gaussianStd = handles.framesFilterParameters(handles.currentFrame).logStd;

% Gettign the frames features.
[~,features,~,idxs] = handles.videoTracker.extractFeatures2(handles.currentFrame, gaussianStd);

% Classifiy correct animals shapes.
acceptence = handles.videoTracker.classify2(features);

% Mark identified animals.
updateFilteredFrame(hObject,idxs(acceptence));


% --- Executes on button press in trackAnimalsButton.
function trackAnimalsButton_Callback(hObject, eventdata, handles)
% hObject    handle to trackAnimalsButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
logStds = [handles.framesFilterParameters.logStd];

if (logStds(1) == 0)
    msgbox('You have to set the filter parameter for at least the first frame.');
    return;
end


for i=1:length(logStds)
    if (logStds(i) == 0)
        if (i > 1)
            logStds(i) = logStds(i-1);
        end
    end
    
end

% Performing tracking.
log(handles,'Tracking in Progress.');
pause(0.1);
handles.videoTracker.performTracking3(handles.allFeatures, logStds);
log(handles,'Tracking completed.');

% Enabling the tracks saving.
if (~isempty(handles.videoTracker.tracks))
    set(handles.saveTracksMenu,'Enable','on');
    set(handles.showTracksButton,'Enable','on');
end


% --------------------------------------------------------------------
function saveTracksMenu_Callback(hObject, eventdata, handles)
% hObject    handle to saveTracksMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fileName = ['Tracks-' strrep(datestr(now),' ','-') '.mat'];

[file,path] = uiputfile(fileName,'Save file name');

% Getting the full path for saving.
fullFileToSave = fullfile(path,file);

% Actually saving the file.
tracks = handles.videoTracker.tracks;
tracker = handles.videoTracker;
save(fullFileToSave,'tracks','tracker');
log(handles,['Saving ' fullFileToSave]);


% --------------------------------------------------------------------
function saveStatisticsMenu_Callback(hObject, eventdata, handles)
% hObject    handle to saveStatisticsMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Getting the information we want to save.
logStds = [handles.framesFilterParameters.logStd];
allFeatures = handles.allFeatures;

% Preparing the default file name.
[videoPath, videoFilename,~] = fileparts(handles.fullFileName);

fileName = [videoFilename '-Features.mat'];
[file,path] = uiputfile(fullfile(videoPath,fileName),'Save file name');

% Getting the full path for saving.
fullFileToSave = fullfile(path,file);

% Actually saving the file.
save(fullFileToSave,'logStds','allFeatures');


% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton11.
function pushbutton11_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on mouse press over axes background.
function filteredFrame_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to filteredFrame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
eventdata


% --- Executes on selection change in logBox.
function logBox_Callback(hObject, eventdata, handles)
% hObject    handle to logBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns logBox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from logBox


% --- Executes during object creation, after setting all properties.
function logBox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to logBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function log(handles,msg)
    logs = get(handles.logBox,'string');
    set(handles.logBox,'string',[logs;msg]);
    set(handles.logBox,'Listboxtop',size(logs,1));


% --- Executes on button press in saveForAllFrames.
function saveForAllFrames_Callback(hObject, eventdata, handles)
% hObject    handle to saveForAllFrames (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
allFrames = 1:handles.videoTracker.numberOfFrames;
handles.framesFilterParameters(allFrames) = repmat(handles.currentFilterParameters, length(allFrames),1);
set(handles.markWormsButton,'Enable','on');

% Forcing the magnigied frame to update.
handles.magnifiedFrameNumber = [];

% Updating handles.
guidata(hObject, handles);

if (~isempty(handles.allFeatures))
    set(handles.showIdenButton,'Enable','on');
else
    set(handles.showIdenButton,'Enable','off');
end


% --- Executes on button press in showTracksButton.
function showTracksButton_Callback(hObject, eventdata, handles)
% hObject    handle to showTracksButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
tracks = handles.videoTracker.tracks;
handles.videoTracker.viewTracks(tracks,[1 handles.videoTracker.numberOfFrames]);
