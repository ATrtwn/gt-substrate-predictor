function varargout = PredictEnzymeInteraction(varargin)
% PREDICTENZYMEINTERACTION M-file for PredictEnzymeInteraction.fig
%      PREDICTENZYMEINTERACTION, by itself, creates a new PREDICTENZYMEINTERACTION or raises the existing
%      singleton*.
%
%      H = PREDICTENZYMEINTERACTION returns the handle to a new PREDICTENZYMEINTERACTION or the handle to
%      the existing singleton*.
%
%      PREDICTENZYMEINTERACTION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PREDICTENZYMEINTERACTION.M with the given input arguments.
%
%      PREDICTENZYMEINTERACTION('Property','Value',...) creates a new PREDICTENZYMEINTERACTION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before PredictEnzymeInteraction_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to PredictEnzymeInteraction_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Copyright 2002-2003 The MathWorks, Inc.

% Edit the above text to modify the response to help PredictEnzymeInteraction

% Last Modified by GUIDE v2.5 17-Aug-2006 10:11:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @PredictEnzymeInteraction_OpeningFcn, ...
                   'gui_OutputFcn',  @PredictEnzymeInteraction_OutputFcn, ...
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


% --- Executes just before PredictEnzymeInteraction is made visible.
function PredictEnzymeInteraction_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to PredictEnzymeInteraction (see VARARGIN)

% Choose default command line output for PredictEnzymeInteraction
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes PredictEnzymeInteraction wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = PredictEnzymeInteraction_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function interactionFileEdit_Callback(hObject, eventdata, handles)
% hObject    handle to interactionFileEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of interactionFileEdit as text
%        str2double(get(hObject,'String')) returns contents of interactionFileEdit as a double


% --- Executes during object creation, after setting all properties.
function interactionFileEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to interactionFileEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in interactionFileButton.
function interactionFileButton_Callback(hObject, eventdata, handles)
% hObject    handle to interactionFileButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


[filename, pathname] = uigetfile('*.*');
file= fullfile(pathname, filename);
if ~isequal(file, 0)   
        set(handles.interactionFileEdit, 'String', file);
end


function protSeqFileEdit_Callback(hObject, eventdata, handles)
% hObject    handle to protSeqFileEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of protSeqFileEdit as text
%        str2double(get(hObject,'String')) returns contents of protSeqFileEdit as a double


% --- Executes during object creation, after setting all properties.
function protSeqFileEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to protSeqFileEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in proteinSeqFileButton.
function proteinSeqFileButton_Callback(hObject, eventdata, handles)
% hObject    handle to proteinSeqFileButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[filename, pathname] = uigetfile('*.*');
file= fullfile(pathname, filename);
if ~isequal(file, 0)   
        set(handles.protSeqFileEdit, 'String', file);
end

function newSeqEdit_Callback(hObject, eventdata, handles)
% hObject    handle to newSeqEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of newSeqEdit as text
%        str2double(get(hObject,'String')) returns contents of newSeqEdit as a double


% --- Executes during object creation, after setting all properties.
function newSeqEdit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to newSeqEdit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in newSeqButton.
function newSeqButton_Callback(hObject, eventdata, handles)
% hObject    handle to newSeqButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname] = uigetfile('*.*');
file= fullfile(pathname, filename);
if ~isequal(file, 0)   
        set(handles.newSeqEdit, 'String', file);
end



% --- Executes on button press in predictButton.
function predictButton_Callback(hObject, eventdata, handles)
% hObject    handle to predictButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% load the data from the files in to the correct format
interactionFile = get(handles.interactionFileEdit , 'String');
seqsFile = get(handles.protSeqFileEdit , 'String');
newSeqFile = get(handles.newSeqEdit , 'String');

Mols = read_enzyme_interaction_file(interactionFile);
[numM, numE] = size(Mols.X);

PS = fastaread(seqsFile);
numPS = length(PS);

newPS = fastaread(newSeqFile);
numNewPS = length(newPS);

%% check that have data for all the enzymes that have fasta sequences for -
%% or just go down list and find nearest neighbour that do have sequences
%% for??

Map = get_Fasta_Data_Map(PS, Mols.Enames);

% find the nearest neighbour using swalign
 for i =1:numPS,
          sw = swalign(PS(i).Sequence, newPS(1).Sequence);
          S(i) = sw;
 end

 [sorted, inds] = sort(S, 'descend');
 enz_ind = find_indexes(Map, inds(1))  % get the interaction data index for the most similar protein
 
 
 if (enz_ind > 0)
 
 %% Go through the data in Mols.X(:, enz_ind) and make a prediction of
 %% either 'yes', 'no' or preidction. State which enzyme is used for the
 %% prediction of each interaction.

 Pred{1,1} = ['Using nearest protein ', char(Mols.Enames{enz_ind(1)}), ' for prediction (Smith-Waterman alignment score ', num2str(S(inds(1))), ')' ];
 Pred{1,2} = [''];
 names = Mols.names{1};
 
 counter =3;
 for m=1:numM,
    s_next = 'Not found';
    val = Mols.X(m, enz_ind);
    switch val 
        case 0 
            class = 'No';
            s_next = [char(names(m)), '     ', class];
            Pred{counter,1} = s_next;
            counter = counter+1;
            fprintf(s_next);
            fprintf('\n');
        case 1 
            class = 'Yes';
            %s_next = [char(Mols.names{m}), '                 ', class];
            s_next = [char(names(m)), '     ', class];
            Pred{counter,1} = s_next;
            counter = counter+1;
            fprintf(s_next);
            fprintf('\n');
    end
 end

  enz_ind2 = find_indexes(Map, inds(2));
  
  if (enz_ind2 > 0)
         Pred{counter,2} = [''];
         Pred{counter+1,1} = ['Using second nearest protein ', char(Mols.Enames{enz_ind2}), ' for prediction (Smith-Waterman alignment score ', num2str(S(inds(2))), ')' ];
         Pred{counter+2,2} = [''];
         counter  = counter +3;
 
              for m=1:numM,
                s_next = 'Not found';
                val = Mols.X(m, enz_ind);
                new_val = Mols.X(m, enz_ind2);
                switch val 
                    case 2
                        switch new_val
                          case 0 
                            class = 'No';
                            s_next = [char(names(m)), '     ', class];
                            Pred{counter,1} = s_next;
                            counter = counter+1;
                            fprintf(s_next);
                            fprintf('\n');
                          case 1 
                            class = 'Yes';
                            %s_next = [char(Mols.names{m}), '                 ', class];
                            s_next = [char(names(m)), '     ', class];
                            Pred{counter,1} = s_next;
                            counter = counter+1;
                            fprintf(s_next);
                            fprintf('\n');
                         end             
                    case 3
                        switch new_val
                          case 0 
                            class = 'No';
                            s_next = [char(names(m)), '     ', class];
                            Pred{counter,1} = s_next;
                            counter = counter+1;
                            fprintf(s_next);
                            fprintf('\n');
                          case 1 
                            class = 'Yes';
                            %s_next = [char(Mols.names{m}), '                 ', class];
                            s_next = [char(names(m)), '     ', class];
                            Pred{counter,1} = s_next;
                            counter = counter+1;
                            fprintf(s_next);
                            fprintf('\n');
                        end             
                end 
            end
 
  end
 displayPrediction(Pred);
 

 else
     msgbox(['No interaction data was found for the closest protein ', char( PS(inds(1)).Header ) ],'Unable to make prediction','modal')
 end
 
% Show the alignment of the new enzyme with the one used for prediction
