from XBrainLab import Study
from XBrainLab import preprocessor
from XBrainLab.ui import XBrainLab
from XBrainLab.visualization import PlotType, VisualizerType
from XBrainLab.dataset import SplitUnit, TrainingType, SplitByType, ValSplitByType
from XBrainLab.load_data import Raw
from XBrainLab.evaluation import Metric
from XBrainLab.load_data import EventLoader
import numpy as np
import torch
from XBrainLab.training import TrainingOption
from XBrainLab.training import TRAINING_EVALUATION
import mne
from XBrainLab.dataset import DataSplitter, DataSplittingConfig
from XBrainLab.training import ModelHolder
from XBrainLab import model_base

study = Study()
study = Study()
lab = XBrainLab(study)
data_loader = study.get_raw_data_loader()


filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0101T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0102T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0103T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0104E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0105E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0201T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0202T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0203T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0204E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0205E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0101T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0101T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('01')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0102T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0102T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('01')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0103T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0103T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('01')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0104E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0104E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('01')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0105E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0105E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('01')
raw_data.set_session_name('5E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0201T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0201T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('02')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0202T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0202T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('02')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0203T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0203T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('02')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0203T.gdf')
raw_data.get_event_list()

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0204E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0204E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('02')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0205E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0205E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('02')
raw_data.set_session_name('5E')

data_loader.validate()

data_loader.apply(study)
selected_channels=['EEG:C3', 'EEG:Cz', 'EEG:C4']
study.preprocess(preprocessor=preprocessor.ChannelSelection, selected_channels=selected_channels)

l_freq=0.5
h_freq=62.5
study.preprocess(preprocessor=preprocessor.Filtering, l_freq=l_freq, h_freq=h_freq)

sfreq=125.0
study.preprocess(preprocessor=preprocessor.Resample, sfreq=sfreq)

selected_event_names=['770', '769', '783']
baseline=None
tmin=-0.5
tmax=3.0
study.preprocess(preprocessor=preprocessor.TimeEpoch, baseline=baseline, selected_event_names=selected_event_names, tmin=tmin, tmax=tmax)

new_event_name={'2': 'RH', '1': 'LH'}
study.preprocess(preprocessor=preprocessor.EditEventName, new_event_name=new_event_name)

test_splitter_list = [
DataSplitter(split_type=SplitByType.SESSION, value_var='0.2', split_unit=SplitUnit.RATIO),
]
val_splitter_list = [
DataSplitter(split_type=ValSplitByType.TRIAL, value_var='0.2', split_unit=SplitUnit.RATIO),
]
datasets_config = DataSplittingConfig(train_type=TrainingType.FULL, is_cross_validation=False, val_splitter_list=val_splitter_list, test_splitter_list=test_splitter_list)
datasets_generator = study.get_datasets_generator(config=datasets_config)


datasets_generator.apply(study)
model_holder = ModelHolder(target_model=model_base.EEGNet, model_params_map={'F1': 8, 'F2': 16, 'D': 2}, pretrained_weight_path=None)

study.set_model_holder(model_holder)
output_dir='/Users/89e/Downloads/DL for biomedical'
optim=torch.optim.Adam
optim_params={'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0, 'amsgrad': False, 'maximize': False, 'capturable': False, 'differentiable': False}
use_cpu=True
gpu_idx=None
epoch=5
bs=32
lr=0.0001
checkpoint_epoch=1
evaluation_option=TRAINING_EVALUATION.VAL_LOSS
repeat_num=1
training_option = TrainingOption(output_dir=output_dir, 
optim=optim, optim_params=optim_params, 
use_cpu=use_cpu, gpu_idx=gpu_idx, 
epoch=epoch, 
bs=bs, 
lr=lr, 
checkpoint_epoch=checkpoint_epoch, 
evaluation_option=evaluation_option, 
repeat_num=repeat_num)

study.set_training_option(training_option)

study.generate_plan()
study.train(interact=True)

study.clean_datasets(force_update=True)
study.reset_preprocess()
study.clean_raw_data(force_update=True)
data_loader = study.get_raw_data_loader()


filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0101T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0102T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0103T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0104E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0105E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0201T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0202T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0203T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0204E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0205E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0301T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0302T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0303T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0304E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0305E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0401T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0402T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0403T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0404E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0405E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0501T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0502T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0503T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0504E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0505E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0601T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0602T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0603T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0604E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0605E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0701T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0702T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0703T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0704E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0705E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0801T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0802T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0803T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0804E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0805E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0901T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0902T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0903T.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0904E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

filepath = '/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0905E.gdf'
data = mne.io.read_raw_gdf(filepath, preload=True)
raw_data = Raw(filepath, data)
data_loader.append(raw_data)

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0101T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0101T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('01')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0102T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0102T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('01')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0103T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0103T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('01')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0104E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0104E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('01')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0105E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0105E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('01')
raw_data.set_session_name('5E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0201T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0201T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('02')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0202T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0202T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('02')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0203T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0203T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('02')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0204E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0204E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('02')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0205E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0205E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('02')
raw_data.set_session_name('5E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0301T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0301T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('03')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0302T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0302T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('03')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0303T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0303T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('03')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0304E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0304E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('03')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0305E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0305E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('03')
raw_data.set_session_name('5E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0401T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0401T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('04')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0402T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0402T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('04')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0403T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0403T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('04')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0404E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0404E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('04')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0405E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0405E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('04')
raw_data.set_session_name('5E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0501T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0501T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('05')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0502T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0502T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('05')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0503T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0503T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('05')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0504E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0504E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('05')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0505E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0505E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('05')
raw_data.set_session_name('5E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0601T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0601T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('06')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0602T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0602T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('06')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0603T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0603T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('06')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0604E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0604E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('06')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0605E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0605E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('06')
raw_data.set_session_name('5E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0701T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0701T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('07')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0702T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0702T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('07')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0703T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0703T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('07')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0704E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0704E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('07')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0705E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0705E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('07')
raw_data.set_session_name('5E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0801T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0801T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('08')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0802T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0802T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('08')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0803T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0803T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('08')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0804E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0804E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('08')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0805E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0805E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('08')
raw_data.set_session_name('5E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0901T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0901T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('09')
raw_data.set_session_name('1T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0902T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0902T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('09')
raw_data.set_session_name('2T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0903T.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0903T.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('09')
raw_data.set_session_name('3T')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0904E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0904E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('09')
raw_data.set_session_name('4E')

raw_data = data_loader.get_loaded_raw('/Users/89e/Downloads/DL for biomedical/BCICIV_2b_gdf/B0905E.gdf')
raw_data.get_event_list()
event_loader = EventLoader(raw_data)
event_loader.read_mat('/Users/89e/Downloads/DL for biomedical/true_labels_2b/B0905E.mat')
event_loader.create_event(event_name_map={1: '1', 2: '2'})
event_loader.apply()
raw_data.set_subject_name('09')
raw_data.set_session_name('5E')

data_loader.validate()

data_loader.apply(study)
selected_channels=['EEG:C3', 'EEG:Cz', 'EEG:C4']
study.preprocess(preprocessor=preprocessor.ChannelSelection, selected_channels=selected_channels)

l_freq=0.5
h_freq=62.5
study.preprocess(preprocessor=preprocessor.Filtering, l_freq=l_freq, h_freq=h_freq)

sfreq=125.0
study.preprocess(preprocessor=preprocessor.Resample, sfreq=sfreq)

selected_event_names=['770', '769', '783']
baseline=None
tmin=-0.5
tmax=3.0
study.preprocess(preprocessor=preprocessor.TimeEpoch, baseline=baseline, selected_event_names=selected_event_names, tmin=tmin, tmax=tmax)

new_event_name={'2': 'RH', '1': 'LH'}
study.preprocess(preprocessor=preprocessor.EditEventName, new_event_name=new_event_name)

test_splitter_list = [
DataSplitter(split_type=SplitByType.SESSION, value_var='0.2', split_unit=SplitUnit.RATIO),
]
val_splitter_list = [
DataSplitter(split_type=ValSplitByType.TRIAL, value_var='0.2', split_unit=SplitUnit.RATIO),
]
datasets_config = DataSplittingConfig(train_type=TrainingType.FULL, is_cross_validation=False, val_splitter_list=val_splitter_list, test_splitter_list=test_splitter_list)
datasets_generator = study.get_datasets_generator(config=datasets_config)


datasets_generator.apply(study)
study.clean_datasets(force_update=True)
test_splitter_list = [
DataSplitter(split_type=SplitByType.SESSION, value_var='0.2', split_unit=SplitUnit.RATIO),
]
val_splitter_list = [
DataSplitter(split_type=ValSplitByType.TRIAL, value_var='0.2', split_unit=SplitUnit.RATIO),
]
datasets_config = DataSplittingConfig(train_type=TrainingType.FULL, is_cross_validation=False, val_splitter_list=val_splitter_list, test_splitter_list=test_splitter_list)
datasets_generator = study.get_datasets_generator(config=datasets_config)


datasets_generator.apply(study)
model_holder = ModelHolder(target_model=model_base.EEGNet, model_params_map={'F1': 8, 'F2': 16, 'D': 2}, pretrained_weight_path=None)

study.set_model_holder(model_holder)
output_dir='/Users/89e/Downloads/DL for biomedical'
optim=torch.optim.Adam
optim_params={'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0, 'amsgrad': False, 'maximize': False, 'capturable': False, 'differentiable': False}
use_cpu=True
gpu_idx=None
epoch=380
bs=32
lr=0.0001
checkpoint_epoch=1
evaluation_option=TRAINING_EVALUATION.VAL_LOSS
repeat_num=1
training_option = TrainingOption(output_dir=output_dir, 
optim=optim, optim_params=optim_params, 
use_cpu=use_cpu, gpu_idx=gpu_idx, 
epoch=epoch, 
bs=bs, 
lr=lr, 
checkpoint_epoch=checkpoint_epoch, 
evaluation_option=evaluation_option, 
repeat_num=repeat_num)

study.set_training_option(training_option)

study.generate_plan()
study.train(interact=True)

lab.show_plot(plot_type=PlotType.CONFUSION, plan_name='0-Group_1', real_plan_name='Repeat-0')
lab.show_performance(metric=Metric.AUC)
lab.show_performance(metric=Metric.KAPPA)

filepath='/Users/89e/Downloads/DL for biomedical/0-Group_1/Repeat-0/0-Group_1-Repeat-0.csv'
plan_name='0-Group_1'
real_plan_name='Repeat-0'
study.export_output_csv(filepath, plan_name, real_plan_name)

montage = mne.channels.make_standard_montage('standard_1005')
chs = ['C3', 'Cz', 'C4']
positions = np.array([montage.get_positions()['ch_pos'][ch] for ch in chs])

study.set_channels(chs, positions)
lab.show_grad_plot(plot_type=VisualizerType.SaliencyMap, plan_name='0-Group_1', real_plan_name='Repeat-0', absolute=False)
lab.show_grad_topo_plot(plot_type=VisualizerType.SaliencyTopoMap, plan_name='0-Group_1', real_plan_name='Repeat-0', absolute=False)
lab.show_grad_eval_plot(plot_type=VisualizerType.SaliencySpectrogramMap, plan_name='0-Group_1', real_plan_name='Repeat-0')
lab.show_plot(plot_type=PlotType.CONFUSION, plan_name='0-Group_1', real_plan_name='Repeat-0')
lab.show_plot(plot_type=PlotType.LOSS, plan_name='0-Group_1', real_plan_name='Repeat-0')
lab.show_plot(plot_type=PlotType.ACCURACY, plan_name='0-Group_1', real_plan_name='Repeat-0')
lab.show_plot(plot_type=PlotType.AUC, plan_name='0-Group_1', real_plan_name='Repeat-0')
lab.show_plot(plot_type=PlotType.LR, plan_name='0-Group_1', real_plan_name='Repeat-0')
