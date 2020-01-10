%% Read in stim file to get the current action word

filepath = fullfile(filesep,'Users','bolger','Documents','work','Projects','Project-EEG-VR','Model-movement-complexity',filesep);
filenom_txt = 's05A_stimdata.txt';
fileall_stim = [filepath,filenom_txt];
fID = fopen(fileall_stim,'r');
StimIn = textscan(fID,'%d %s %s %s %s %s %s','CommentStyle','//');

igo = find(strcmp(string(StimIn{1,5}),'GO'));
inogo = find(strcmp(string(StimIn{1,5}),'NOGO'));


xlIn = dir(fullfile(filepath,'*.xlsx'));
fileInxls = {xlIn.name};
fileInxls = fileInxls(1,1:length(StimIn{1,1}));

%% Read in the trial level files that need to be in excel format.
f1= figure; set(f1,'Color',[1 1 1]);
cols = 8;
rows = ceil(length(igo)/cols);

movement_features = cell(length(igo),4);

for fcnt = 1:length(igo)
    
    display(strcat(string(StimIn{1,2}(igo(fcnt))), '-' ,string(StimIn{1,5}(igo(fcnt))),'-',string(StimIn{1,6}(igo(fcnt)))));
    
        
        t = ['Trial_Order',num2str(igo(fcnt)-1),'_'];
        X = strfind(string(fileInxls),t);
        x_temp = cell2mat(cellfun(@isempty,X,'UniformOutput',false));
        findx = find(x_temp==0);

        filexls_curr = [filepath,fileInxls{1,findx}];

        [~, txt, raw] = xlsread(filexls_curr,1,'A1:M11266');
        rownum = size(txt,1);
        colnum = size(txt,2);
        headers = string(txt(1,:));     %cell array of header titles
        numpart = txt(2:size(txt,1),1:size(txt,2));

        i = find((strcmp(headers,'Phase')));
        ptype = string(numpart(2:rownum-1,i));
        phasetype = unique(ptype);

        %% Find the phase corresponding to Action
        indxp = find(strcmp(ptype,'Action'));

        indxhposx = find(strcmp(headers,'hand_posx'));
        indxhposy = find(strcmp(headers,'hand_posy'));
        indxhposz = find(strcmp(headers,'hand_posz'));
        indxtime = find(strcmp(headers,'TryTime'));

        handposx_all = numpart(2:rownum-1,indxhposx);
        handposy_all = numpart(2:rownum-1,indxhposy);
        handposz_all = numpart(2:rownum-1,indxhposz);
        time_all = numpart(2:rownum-1,indxtime);

        handposx_act = handposx_all(indxp,1);
        handposy_act = handposy_all(indxp,1);
        handposz_act = handposz_all(indxp,1);
        time_act = time_all(indxp,1);

        handposx_act = cell2mat(cellfun(@str2double,handposx_act,'UniformOutput',false));
        handposy_act = cell2mat(cellfun(@str2double,handposy_act,'UniformOutput',false));
        handposz_act = cell2mat(cellfun(@str2double,handposz_act,'UniformOutput',false));
        time_act = cell2mat(cellfun(@str2double,time_act,'UniformOutput',false));
        
       
        handgradx_act = gradient(handposx_act);
        handgrady_act = gradient(handposy_act);
        handgradz_act = gradient(handposz_act);
        
        [Velocity_act,Trs] = velocity_calc([handposx_act,handposy_act, handposz_act],time_act);     % Call of function to calculate the velocity
        Accel_act = accel_calc(Velocity_act,Trs);
        [TAngle_act,timers_act] = turnangle_calc([handposx_act,handposy_act, handposz_act],time_act);
        
        movement_features{fcnt,1} = Velocity_act;
        movement_features{fcnt,2} = Accel_act;
        movement_features{fcnt,3} = TAngle_act;
        
        subplot(rows,cols,fcnt)
        quiver3(handposx_act,handposy_act,handposz_act,handgradx_act,handgrady_act,handgradz_act,0.5)
        view(-51,32)
        title(strcat(string(StimIn{1,2}(igo(fcnt))), '-' ,string(StimIn{1,5}(igo(fcnt))),'-',string(StimIn{1,6}(igo(fcnt)))));
        
        axhdl = gca;
        titre_curr = strcat(string(StimIn{1,2}(igo(fcnt))), '-' ,string(StimIn{1,5}(igo(fcnt))),'-',string(StimIn{1,6}(igo(fcnt))));
        set(f1,'CurrentAxes',axhdl);
        set(axhdl,'HitTest','on','SelectionHighlight','on','UserData',{movement_features{fcnt,1},movement_features{fcnt,2},movement_features{fcnt,3},...
            time_act,titre_curr,timers_act});
        set(axhdl,'ButtonDownFcn',@plotsingle_movement_features)
  
end


