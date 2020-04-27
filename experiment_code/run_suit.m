function varargout=run_suit(what,varargin)

%========================================================================================================================
%% (1) Directories
BASE_DIR          = '/global/scratch/maedbhking/projects/cerebellum_language'; 

DATA_DIR          = fullfile(BASE_DIR, 'data'); 
FMRIPREP_DIR      = fullfile(DATA_DIR, 'fmriprep_output/fmriprep'); 
SUIT_DIR          = fullfile(DATA_DIR, 'suit');

subj_names = {'sub-01', 'sub-02', 'sub-03'}; 

%% add software to search path
addpath(genpath('/global/home/users/maedbhking/spm12'))

%% run suit
run_suit('SUIT:run_all')

%% functions

switch(what)
    case 'SUIT:run_all'                      % STEP 9.1-8.5
        glm = 1; % glmNum
        spm fmri
        
        for s=1:length(subj_names),
            run_suit('SUIT:isolate_segment',s)
%             run_suit('SUIT:make_mask_cortex',s)
%             run_suit('SUIT:corr_cereb_cortex_mask',s)
%             run_suit('SUIT:normalise_dartel',s,'grey');
%             run_suit('SUIT:make_mask',s,glm,'grey');
%             run_suit('SUIT:reslice',s,glm,'contrast','cereb_prob_corr_grey');
%             run_suit('SUIT:reslice',s,1,glm,'ResMS','cereb_prob_corr_grey');
%             run_suit('SUIT:reslice',s,2,glm,'ResMS','cereb_prob_corr_grey');
%             run_suit('SUIT:reslice',s,1,glm,'betas','cereb_prob_corr_grey');
%             run_suit('SUIT:reslice',s,2,glm,'betas','cereb_prob_corr_grey');

        end
    case 'SUIT:isolate_segment'              % STEP 9.2:Segment cerebellum into grey and white matter
        sn=varargin{1};
        %         spm fmri
        for s=sn
            anat_name = sprintf('%s_desc-preproc_T1w.nii.gz',subj_names{s});
            SUIT_SUBJ_DIR = fullfile(SUIT_DIR, 'anatomicals',subj_names{s});dircheck(SUIT_SUBJ_DIR);
            SOURCE=fullfile(FMRIPREP_DIR, subj_names{s}, 'anat', anat_name); 
            DEST=fullfile(SUIT_SUBJ_DIR, anat_name);
            copyfile(SOURCE, DEST);
            cd(fullfile(SUIT_SUBJ_DIR));
            suit_isolate_seg({fullfile(SUIT_SUBJ_DIR, anat_name)},'keeptempfiles',1);
        end
    case 'SUIT:make_mask_cortex'             % STEP 9.3:
        % STUDY 1
        sn=varargin{1};
        
        subjs=length(sn);
        for s=1:subjs,
            glmSubjDir =[studyDir{1} '/GLM_firstlevel_4/' subj_names{sn(s)}];
            
            for h=regSide,
                C=caret_load(fullfile(studyDir{1},caretDir,atlasname,hemName{h},[hem{h} '.cerebral_cortex.paint'])); % freesurfer
                caretSubjDir=fullfile(studyDir{1},caretDir,[atlasA subj_names{sn(s)}]);
                file=fullfile(glmSubjDir,'mask.nii');
                r{h}.type='surf_nodes';
                r{h}.location=find(C.data(:,1)~=6); % we don't want the medial wall
                r{h}.white=fullfile(caretSubjDir,hemName{h},[hem{h} '.WHITE.coord']);
                r{h}.pial=fullfile(caretSubjDir,hemName{h},[hem{h} '.PIAL.coord']);
                r{h}.topo=fullfile(caretSubjDir,hemName{h},[hem{h} '.CLOSED.topo']);
                r{h}.linedef=[5,0,1];
                r{h}.image=file;
                r{h}.file=file;
                r{h}.name=[hem{h}];
            end;
            r=region_calcregions(r);
            r{1}.location=vertcat(r{1}.location,r{2}.location);
            r{1}.linvoxidxs=vertcat(r{1}.linvoxidxs,r{2}.linvoxidxs);
            r{1}.data=vertcat(r{1}.data,r{2}.data);
            r{1}.weight=vertcat(r{1}.weight,r{2}.weight);
            r{1}.name='cortical_mask_grey';
            R{1}=r{1};
            dircheck(fullfile(studyDir{1},regDir,'data',subj_names{sn(s)}));
            cd(fullfile(studyDir{1},regDir,'data',subj_names{sn(s)}))
            region_saveasimg(R{1},R{1}.file);
        end
    case 'SUIT:corr_cereb_cortex_mask'       % STEP 9.4:
        sn=varargin{1};
        % STUDY 1
        
        subjs=length(sn);
        
        for s=1:subjs,
            
            cortexGrey= fullfile(studyDir{1},regDir,'data',subj_names{sn(s)},'cortical_mask_grey.nii'); % cerebellar mask grey (corrected)
            cerebGrey = fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)},'c1anatomical.nii'); % was 'cereb_prob_corr_grey.nii'
            bufferVox = fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)},'buffer_voxels.nii');
            
            % isolate overlapping voxels
            spm_imcalc({cortexGrey,cerebGrey},bufferVox,'(i1.*i2)')
            
            % mask buffer
            spm_imcalc({bufferVox},bufferVox,'i1>0')
            
            cerebGrey2 = fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)},'cereb_prob_corr_grey.nii');
            cortexGrey2= fullfile(studyDir{1},regDir,'data',subj_names{sn(s)},'cortical_mask_grey_corr.nii');
            
            % remove buffer from cerebellum
            spm_imcalc({cerebGrey,bufferVox},cerebGrey2,'i1-i2')
            
            % remove buffer from cortex
            spm_imcalc({cortexGrey,bufferVox},cortexGrey2,'i1-i2')
        end
    case 'SUIT:normalise_dartel'             % STEP 9.5: Normalise the cerebellum into the SUIT template.
        % STUDY 1
        % Normalise an individual cerebellum into the SUIT atlas template
        % Dartel normalises the tissue segmentation maps produced by suit_isolate
        % to the SUIT template
        % !! Make sure that you're choosing the correct isolation mask
        % (corr OR corr1 OR corr2 etc)!!
        % if you are running multiple subjs - change to 'job.subjND(s)."'
        % example: sc1_sc2_imana('SUIT:normalise_dartel',1,'grey')
        sn=varargin{1}; %subjNum
        type=varargin{2}; % 'grey' or 'whole' cerebellar mask
        
        subjs=length(sn);
        for s=1:subjs,
            cd(fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)}));
            job.subjND.gray      = {'a_c_anatomical_seg1.nii'};
            job.subjND.white     = {'a_c_anatomical_seg2.nii'};
            switch type,
                case 'grey'
                    job.subjND.isolation= {'cereb_prob_corr_grey.nii'};
                case 'whole'
                    job.subjND.isolation= {'cereb_prob_corr.nii'};
            end
            suit_normalize_dartel(job);
        end
        
        % 'spm_dartel_warp' code was changed to look in the working
        % directory for 'u_a_anatomical_segment1.nii' file - previously it
        % was giving a 'file2mat' error because it mistakenly believed that
        % this file had been created
    case 'SUIT:normalise_dentate'            % STEP 9.6: Uses an ROI from the dentate nucleus to improve the overlap of the DCN
        % STUDY 1
        sn=varargin{1}; %subjNum
        type=varargin{2}; % 'grey' or 'whole'
        % example: 'sc1_sc2_imana('SUIT:normalise_dentate',2,'grey'
        
        cd(fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn}));
        job.subjND.gray       = {'c_anatomical_seg1.nii'};
        job.subjND.white      = {'c_anatomical_seg2.nii'};
        job.subjND.dentateROI = {fullfile(studyDir{1},suitDir,'glm4',subj_names{sn},'dentate_mask.nii')};
        switch type,
            case 'grey'
                job.subjND.isolation  = {'cereb_prob_corr_grey.nii'};
            case 'whole'
                job.subjND.isolation  = {'cereb_prob_corr.nii'};
        end
        
        suit_normalize_dentate(job);
    case 'SUIT:make_mask'                    % STEP 9.7: Make cerebellar mask using SUIT
        % STUDY 1 ONLY
        % example: sc1_sc2_imana('SUIT:make_mask',1,4,'grey')
        sn=varargin{1}; % subjNum
        glm=varargin{2}; % glmNum
        type=varargin{3}; % 'grey' or 'whole'
        
        subjs=length(sn);
        
        for s=1:subjs,
            glmSubjDir = fullfile(studyDir{1},sprintf('GLM_firstlevel_%d',glm),subj_names{sn(s)});
            mask       = fullfile(glmSubjDir,'mask.nii'); % mask for functional image
            switch type
                case 'grey'
                    suit  = fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)},'cereb_prob_corr_grey.nii'); % cerebellar mask grey (corrected)
                    omask = fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)},'maskbrainSUITGrey.nii'); % output mask image - grey matter
                case 'whole'
                    suit  = fullfile(studyDir{1},suitDir,'anatomicals', subj_names{sn(s)},'cereb_prob_corr.nii'); % cerebellar mask (corrected)
                    omask = fullfile(studyDir{1},suitDir, sprintf('glm%d',glm),subj_names{sn(s)},'maskbrainSUIT.nii'); % output mask image
            end
            dircheck(fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)}));
            cd(fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)}));
            spm_imcalc({mask,suit},omask,'i1>0 & i2>0.7',{});
        end
    case 'SUIT:reslice'                      % STEP 9.8: Reslice the contrast images from first-level GLM
        % Reslices the functional data (betas, contrast images or ResMS)
        % from the first-level GLM using deformation from
        % 'suit_normalise_dartel'.
        % example: sc1_sc2_imana('SUIT:reslice',1,1,4,'betas','cereb_prob_corr_grey')
        % make sure that you reslice into 2mm^3 resolution
        sn=varargin{1}; % subjNum
        study=varargin{2}; % studyNum
        glm=varargin{3}; % glmNum
        type=varargin{4}; % 'betas' or 'contrast' or 'ResMS' or 'cerebellarGrey'
        mask=varargin{5}; % 'cereb_prob_corr_grey' or 'cereb_prob_corr' or 'dentate_mask'
        
        subjs=length(sn);
        
        for s=1:subjs,
            switch type
                case 'betas'
                    glmSubjDir = fullfile(studyDir{study},sprintf('GLM_firstlevel_%d',glm),subj_names{sn(s)});
                    outDir=fullfile(studyDir{study},suitDir,sprintf('glm%d',glm),subj_names{sn(s)});
                    images='beta_0';
                    source=dir(fullfile(glmSubjDir,sprintf('*%s*',images))); % images to be resliced
                    cd(glmSubjDir);
                case 'contrast'
                    glmSubjDir = fullfile(studyDir{study},sprintf('GLM_firstlevel_%d',glm),subj_names{sn(s)});
                    outDir=fullfile(studyDir{study},suitDir,sprintf('glm%d',glm),subj_names{sn(s)});
                    images='con';
                    source=dir(fullfile(glmSubjDir,sprintf('*%s*',images))); % images to be resliced
                    cd(glmSubjDir);
                case 'ResMS'
                    glmSubjDir = fullfile(studyDir{study},sprintf('GLM_firstlevel_%d',glm),subj_names{sn(s)});
                    outDir=fullfile(studyDir{study},suitDir,sprintf('glm%d',glm),subj_names{sn(s)});
                    images='ResMS';
                    source=dir(fullfile(glmSubjDir,sprintf('*%s*',images))); % images to be resliced
                    cd(glmSubjDir);
                case 'cerebellarGrey'
                    source=dir(fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)},'c1anatomical.nii')); % image to be resliced
                    cd(fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)}));
            end
            job.subj.affineTr = {fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)},'Affine_c_anatomical_seg1.mat')};
            job.subj.flowfield= {fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)},'u_a_c_anatomical_seg1.nii')};
            job.subj.resample = {source.name};
            job.subj.mask     = {fullfile(studyDir{1},suitDir,'anatomicals',subj_names{sn(s)},sprintf('%s.nii',mask))};
            job.vox           = [2 2 2];
            suit_reslice_dartel(job);
            if ~strcmp(type,'cerebellarGrey'),
                source=fullfile(glmSubjDir,'*wd*');
                dircheck(fullfile(outDir));
                destination=fullfile(studyDir{study},suitDir,sprintf('glm%d',glm),subj_names{sn(s)});
                movefile(source,destination);
            end
            fprintf('%s have been resliced into suit space for %s \n\n',type,glm,subj_names{sn(s)})
        end
end

%% Local functions
function dircheck(dir)
if ~exist(dir,'dir');
    warning('%s doesn''t exist. Creating one now. You''re welcome! \n',dir);
    mkdir(dir);
end


