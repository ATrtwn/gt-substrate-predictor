function varargout = Predict_Acceptor_Interaction(varargin)
% PREDICT_ACCEPTOR_INTERACTION M-file for Predict_Acceptor_Interaction.fig
%      PREDICT_ACCEPTOR_INTERACTION, by itself, creates a new PREDICT_ACCEPTOR_INTERACTION or raises the existing
%      singleton*.
%
%      H = PREDICT_ACCEPTOR_INTERACTION returns the handle to a new PREDICT_ACCEPTOR_INTERACTION or the handle to
%      the existing singleton*.
%
%      PREDICT_ACCEPTOR_INTERACTION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PREDICT_ACCEPTOR_INTERACTION.M with the given input arguments.
%
%      PREDICT_ACCEPTOR_INTERACTION('Property','Value',...) creates a new PREDICT_ACCEPTOR_INTERACTION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Predict_Acceptor_Interaction_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Predict_Acceptor_Interaction_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Copyright 2002-2003 The MathWorks, Inc.

% Edit the above text to modify the response to help Predict_Acceptor_Interaction

% Last Modified by GUIDE v2.5 16-Aug-2006 02:56:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Predict_Acceptor_Interaction_OpeningFcn, ...
                   'gui_OutputFcn',  @Predict_Acceptor_Interaction_OutputFcn, ...
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


% --- Executes just before Predict_Acceptor_Interaction is made visible.
function Predict_Acceptor_Interaction_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Predict_Acceptor_Interaction (see VARARGIN)

% Choose default command line output for Predict_Acceptor_Interaction
handles.output = hObject;

% organise the pop-up menu
vals = {'Flavonoid', 'Coumarins', 'Cytokinins', 'Cinnamic acid', 'Benzoate', 'Jasmonic acid', 'Gibberellins', 'Auxin', 'Abscisic acid', 'Sinapic acid', 'Other'};
set(handles.familyPopupMenu, 'String', vals);

handles.Data.groups(1).List = {'F 3-OH', 'F 5-OH', 'F 6-OH', 'F 7-OH', 'F 13-OH', 'F 14-OH'};
handles.Data.groups(2).List = {'Cm 6-OH', 'Cm 7-OH'};
handles.Data.groups(3).List = {'Ck 3-N', 'Ck 7-N', 'Ck -OH'};
handles.Data.groups(4).List = {'Cn 2-OH', 'Cn 3-OH', 'Cn 4-OH'};
handles.Data.groups(5).List = [];
handles.Data.groups(6).List = [];
handles.Data.groups(7).List = [];
handles.Data.groups(8).List = [];
handles.Data.groups(9).List = [];
handles.Data.groups(10).List = [];
handles.Data.groups(11).List = [];

%Data = read_acc_struct_pka_file('acceptors_struct_pka.txt');
%handles.Data  = Data;
%thres = (handles.Data.X(:, [5, 7:20])>0);
%handles.Data.X(:, [5, 7:20]) = thres;
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Predict_Acceptor_Interaction wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Predict_Acceptor_Interaction_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in familyPopupMenu.
function familyPopupMenu_Callback(hObject, eventdata, handles)
% hObject    handle to familyPopupMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns familyPopupMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from familyPopupMenu

val = get(handles.familyPopupMenu, 'Value');
list = handles.Data.groups(val).List;
num = length(list);

switch num 
    case 0
       %  make everything invisible
        set(handles.group1checkbox, 'Visible', 'off');  
        set(handles.group2checkbox, 'Visible', 'off');  
        set(handles.group3checkbox, 'Visible', 'off');  
        set(handles.group4checkbox, 'Visible', 'off');  
        set(handles.group5checkbox, 'Visible', 'off');  
        set(handles.group6checkbox, 'Visible', 'off');  
        set(handles.group1checkbox, 'String', '');  
        set(handles.group2checkbox, 'String', '');  
        set(handles.group3checkbox, 'String', '');  
        set(handles.group4checkbox, 'String', '');  
        set(handles.group5checkbox, 'String', '');  
        set(handles.group6checkbox, 'String', '');
       
    case 2
      % make first two visible - change the labels
        set(handles.group1checkbox, 'Visible', 'on');  
        set(handles.group2checkbox, 'Visible', 'on');  
        set(handles.group3checkbox, 'Visible', 'off');  
        set(handles.group4checkbox, 'Visible', 'off');  
        set(handles.group5checkbox, 'Visible', 'off');  
        set(handles.group6checkbox, 'Visible', 'off');  
        set(handles.group1checkbox, 'String', list(1));  
        set(handles.group2checkbox, 'String', list(2));  
        set(handles.group3checkbox, 'String', '');  
        set(handles.group4checkbox, 'String', '');  
        set(handles.group5checkbox, 'String', '');  
        set(handles.group6checkbox, 'String', '');   
   
    case 3
       % make first three visible - set the correct names
       
        set(handles.group1checkbox, 'Visible', 'on');  
        set(handles.group2checkbox, 'Visible', 'on');  
        set(handles.group3checkbox, 'Visible', 'on');  
        set(handles.group4checkbox, 'Visible', 'off');  
        set(handles.group5checkbox, 'Visible', 'off');  
        set(handles.group6checkbox, 'Visible', 'off');  
        set(handles.group1checkbox, 'String', list(1));  
        set(handles.group2checkbox, 'String', list(2));  
        set(handles.group3checkbox, 'String', list(3));  
        set(handles.group4checkbox, 'String', '');  
        set(handles.group5checkbox, 'String', '');  
        set(handles.group6checkbox, 'String', '');     

    
    case 6
       % make all visisble with the correct names 
       
        set(handles.group1checkbox, 'Visible', 'on');  
        set(handles.group2checkbox, 'Visible', 'on');  
        set(handles.group3checkbox, 'Visible', 'on');  
        set(handles.group4checkbox, 'Visible', 'on');  
        set(handles.group5checkbox, 'Visible', 'on');  
        set(handles.group6checkbox, 'Visible', 'on');  
        set(handles.group1checkbox, 'String', list(1));  
        set(handles.group2checkbox, 'String', list(2));  
        set(handles.group3checkbox, 'String', list(3));  
        set(handles.group4checkbox, 'String', list(4));  
        set(handles.group5checkbox, 'String', list(5));  
        set(handles.group6checkbox, 'String', list(6));    
     
end




% --- Executes during object creation, after setting all properties.
function familyPopupMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to familyPopupMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function numOHedit_Callback(hObject, eventdata, handles)
% hObject    handle to numOHedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of numOHedit as text
%        str2double(get(hObject,'String')) returns contents of numOHedit as a double


% --- Executes during object creation, after setting all properties.
function numOHedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to numOHedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in makePredictionButton.
function makePredictionButton_Callback(hObject, eventdata, handles)
% hObject    handle to makePredictionButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function LogPedit_Callback(hObject, eventdata, handles)
% hObject    handle to LogPedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of LogPedit as text
%        str2double(get(hObject,'String')) returns contents of LogPedit as a double


% --- Executes during object creation, after setting all properties.
function LogPedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to LogPedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function AAreaedit_Callback(hObject, eventdata, handles)
% hObject    handle to AAreaedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of AAreaedit as text
%        str2double(get(hObject,'String')) returns contents of AAreaedit as a double


% --- Executes during object creation, after setting all properties.
function AAreaedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to AAreaedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function voledit_Callback(hObject, eventdata, handles)
% hObject    handle to voledit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of voledit as text
%        str2double(get(hObject,'String')) returns contents of voledit as a double


% --- Executes during object creation, after setting all properties.
function voledit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to voledit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function COOHedit_Callback(hObject, eventdata, handles)
% hObject    handle to COOHedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of COOHedit as text
%        str2double(get(hObject,'String')) returns contents of COOHedit as a double


% --- Executes during object creation, after setting all properties.
function COOHedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to COOHedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function group1edit_Callback(hObject, eventdata, handles)
% hObject    handle to group1edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of group1edit as text
%        str2double(get(hObject,'String')) returns contents of group1edit as a double


% --- Executes during object creation, after setting all properties.
function group1edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to group1edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function group2edit_Callback(hObject, eventdata, handles)
% hObject    handle to group2edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of group2edit as text
%        str2double(get(hObject,'String')) returns contents of group2edit as a double


% --- Executes during object creation, after setting all properties.
function group2edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to group2edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function group3edit_Callback(hObject, eventdata, handles)
% hObject    handle to group3edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of group3edit as text
%        str2double(get(hObject,'String')) returns contents of group3edit as a double


% --- Executes during object creation, after setting all properties.
function group3edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to group3edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function group4edit_Callback(hObject, eventdata, handles)
% hObject    handle to group4edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of group4edit as text
%        str2double(get(hObject,'String')) returns contents of group4edit as a double


% --- Executes during object creation, after setting all properties.
function group4edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to group4edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function group5edit_Callback(hObject, eventdata, handles)
% hObject    handle to group5edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of group5edit as text
%        str2double(get(hObject,'String')) returns contents of group5edit as a double


% --- Executes during object creation, after setting all properties.
function group5edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to group5edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in COOHcheckbox.
function COOHcheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to COOHcheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of COOHcheckbox


% --- Executes on button press in group1checkbox.
function group1checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to group1checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of group1checkbox


% --- Executes on button press in group2checkbox.
function group2checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to group2checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of group2checkbox


% --- Executes on button press in group3checkbox.
function group3checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to group3checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of group3checkbox


% --- Executes on button press in group4checkbox.
function group4checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to group4checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of group4checkbox


% --- Executes on button press in group5checkbox.
function group5checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to group5checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of group5checkbox


% --- Executes on button press in group6checkbox.
function group6checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to group6checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of group6checkbox



function group6edit_Callback(hObject, eventdata, handles)
% hObject    handle to group6edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of group6edit as text
%        str2double(get(hObject,'String')) returns contents of group6edit as a double


% --- Executes during object creation, after setting all properties.
function group6edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to group6edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function Areaedit_Callback(hObject, eventdata, handles)
% hObject    handle to Areaedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Areaedit as text
%        str2double(get(hObject,'String')) returns contents of Areaedit as a double


% --- Executes during object creation, after setting all properties.
function Areaedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Areaedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function logPedit_Callback(hObject, eventdata, handles)
% hObject    handle to logPedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of logPedit as text
%        str2double(get(hObject,'String')) returns contents of logPedit as a double


% --- Executes during object creation, after setting all properties.
function logPedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to logPedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end



function edit15_Callback(hObject, eventdata, handles)
% hObject    handle to voledit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of voledit as text
%        str2double(get(hObject,'String')) returns contents of voledit as a double


% --- Executes during object creation, after setting all properties.
function edit15_CreateFcn(hObject, eventdata, handles)
% hObject    handle to voledit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in predictionButton.
function predictionButton_Callback(hObject, eventdata, handles)
% hObject    handle to predictionButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Data = loadData(handles);
handles.Data = Data;
guidata(hObject, handles);

% get all the values form the fields   
logP = str2num(get(handles.logPedit ,'String'));
vol = str2num(get(handles.voledit ,'String'));
area = str2num(get(handles.Areaedit ,'String'));
family = get(handles.familyPopupMenu, 'Value'); 
numOH =  str2num(get(handles.numOHedit,'String'));

cooh  = get(handles.COOHcheckbox ,'Value');

groups = zeros(1, 14);
switch family 
    case 1
%    % get flav groups => set the pka values
     f1 = get(handles.group1checkbox ,'Value');
     f2 = get(handles.group2checkbox ,'Value');
     f3 = get(handles.group3checkbox ,'Value');
     f4 = get(handles.group4checkbox ,'Value');
     f5 = get(handles.group5checkbox ,'Value');
     f6 = get(handles.group6checkbox ,'Value');
     groups(1:6) = [f1, f2, f3, f4, f5, f6]; 
    case 2
%    % get cm groups
     c1 = get(handles.group1checkbox ,'Value')
     c2 = get(handles.group2checkbox ,'Value')
     groups(7:8) = [c1, c2]; 
    case 3
%    % get ck groups
     c1 = get(handles.group1checkbox ,'Value');
     c2 = get(handles.group2checkbox ,'Value');
     c3 = get(handles.group3checkbox ,'Value'); 
     groups(9:11) = [c1, c2, c3]; 
    case 4
%    % get cn groups
     c1 = get(handles.group1checkbox ,'Value');
     c2 = get(handles.group2checkbox ,'Value');
     c3 = get(handles.group3checkbox ,'Value'); 
     groups(12:14) = [c1, c2, c3]; 
end

X_new = [family, logP, area, vol, cooh, numOH, groups];
%X_new = handles.Data.X(3,:);

% train all of classification trees
[dum, numE] = size(handles.Data.E);
for enz=1:numE,
  L = handles.Data.E(:,enz);
  I = find(L<2);
  t = treefit(handles.Data.X(I,:), L(I) ,'method', 'classification', 'catidx', [1, 5, 7:20], 'splitcriterion', 'deviance', 'splitmin', 1);
  %treedisp(t ,'names', handles.Data.Xnames)
  %[sfit, node, cnames] = treeval(t,handles.Data.X(I,:));      % find assigned class numbers
  %prate = sum(str2num(char(cnames))==L(I))/length(I)*100;   % com
  [sfit, node, class] = treeval(t, X_new);

    cl = 'No';
    if str2num(char(class))==1
        cl = 'Yes';
    end

    if str2num(char(class))==1
    s_next = [char(handles.Data.Enames{enz}), '     ', cl];
    elseif str2num(char(class))==0
    s_next = [char(handles.Data.Enames{enz}), '                 ', cl];
    end

    S{enz,1} = s_next;
    fprintf(s_next);
    fprintf('\n');
end
displayEnzymes(S);




function filenameEdit_Callback(hObject, eventdata, handles)
% hObject    handle to filenameEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of filenameEdit as text
%        str2double(get(hObject,'String')) returns contents of filenameEdit as a double


% --- Executes during object creation, after setting all properties.
function filenameEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to filenameEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in filechooserButton.
function filechooserButton_Callback(hObject, eventdata, handles)
% hObject    handle to filechooserButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[filename, pathname] = uigetfile('*.txt');
file= fullfile(pathname, filename);
if ~isequal(file, 0)   
        set(handles.filenameEdit, 'String', file);
end

% --- Executes on button press in filechooserButton.
function Data = loadData(handles)
filename = get(handles.filenameEdit, 'String');
Data = read_acceptor_interaction_file(filename);
thres = (Data.X(:, [5, 7:20])>0);
Data.X(:, [5, 7:20]) = thres;
