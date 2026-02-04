function varargout = displayPrediction(varargin)
% DISPLAYPREDICTION M-file for displayPrediction.fig
%      DISPLAYPREDICTION, by itself, creates a new DISPLAYPREDICTION or raises the existing
%      singleton*.
%
%      H = DISPLAYPREDICTION returns the handle to a new DISPLAYPREDICTION or the handle to
%      the existing singleton*.
%
%      DISPLAYPREDICTION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DISPLAYPREDICTION.M with the given input arguments.
%
%      DISPLAYPREDICTION('Property','Value',...) creates a new DISPLAYPREDICTION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before displayPrediction_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to displayPrediction_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Copyright 2002-2003 The MathWorks, Inc.

% Edit the above text to modify the response to help displayPrediction

% Last Modified by GUIDE v2.5 23-Apr-2005 20:08:36

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @displayPrediction_OpeningFcn, ...
                   'gui_OutputFcn',  @displayPrediction_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
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

% --- Executes just before displayPrediction is made visible.
function displayPrediction_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to displayPrediction (see VARARGIN)

        handles.genelist = varargin{1};
        [n, m] = size(handles.genelist);
        handles.numgenes = n;
        p = get(handles.genelistEdit, 'Position'); % get the height in characters
        handles.numtodisplay = round(p(4));
        if n<handles.numtodisplay
            handles.numtodisplay = n;
        end
        %set(handles.text, 'String', ['Number of Enzymes: ', num2str(handles.numgenes)]);
        set(handles.genelistEdit, 'String', handles.genelist(1:handles.numtodisplay, :));
        set(handles.scroller, 'Min', 1.0);
        set(handles.scroller, 'Max', handles.numgenes);
        set(handles.scroller, 'Value', handles.numgenes);
        set(handles.scroller, 'SliderStep', [1/n, 0.1])
        % Choose default command line output for displayPrediction
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes displayPrediction wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = displayPrediction_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


function genelistEdit_Callback(hObject, eventdata, handles)
% hObject    handle to genelistEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of genelistEdit as text
%        str2double(get(hObject,'String')) returns contents of genelistEdit as a double


% --- Executes during object creation, after setting all properties.
function genelistEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to genelistEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in closeButton.
function closeButton_Callback(hObject, eventdata, handles)
% hObject    handle to closeButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
delete(handles.figure1);

% --- Executes on slider movement.
function scroller_Callback(hObject, eventdata, handles)
% hObject    handle to scroller (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
 val = round(get(handles.scroller, 'Value'));
 startval = handles.numgenes - val + 1 ;
 endval =  startval + handles.numtodisplay;
 %if startval>(handles.numgenes-handles.numtodisplay)
 %    startval =  handles.numgenes-handles.numtodisplay   
 %    endval = handles.numgenes
 %end
 %if startval <1
 %    startval = 1
 %end
 if endval>(handles.numgenes)
 endval = handles.numgenes ;
 end
 set(handles.genelistEdit, 'String', handles.genelist(startval:endval, :))


% --- Executes during object creation, after setting all properties.
function scroller_CreateFcn(hObject, eventdata, handles)
% hObject    handle to scroller (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background, change
%       'usewhitebg' to 0 to use default.  See ISPC and COMPUTER.
usewhitebg = 1;
if usewhitebg
    set(hObject,'BackgroundColor',[.9 .9 .9]);
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


