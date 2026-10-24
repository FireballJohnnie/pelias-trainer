# provenance: original
# Copyright (c) 2026 John Lipka and the Pelias contributors
#
# project: The one project file, run folders under the user's output folder, the model key.
# pelias-deps: io
pelias_module(project
    CC project_file.cc run_folder.cc model_key.cc
    DEPS io)
