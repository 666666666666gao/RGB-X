function score_visevent_matlab(toolkit_dir, names_file, gt_dir, results_dir, tracker_name, output_dir)
% Execute the unchanged author's scorer on the explicitly frozen annotation version.
addpath(fullfile(toolkit_dir, 'utils'));
save_default_options('-mat7-binary');
file = fopen(names_file, 'r');
sequences = textscan(file, '%s');
fclose(file);
sequences = sequences{1};
trackers = {struct('name', tracker_name)};
names = {tracker_name};
eval_tracker(sequences, trackers, 'OPE', names, [fullfile(output_dir, 'raw') filesep], ...
             [gt_dir filesep], [results_dir filesep], false);
eval_tracker(sequences, trackers, 'OPE', names, [fullfile(output_dir, 'normalized') filesep], ...
             [gt_dir filesep], [results_dir filesep], true);
end
