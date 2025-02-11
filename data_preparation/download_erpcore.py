from osfclient.api import OSF
import os

# Initialize OSF
osf = OSF()

# List of project IDs
projects = [
    # '5q4xs',
    "etdkz",
    # '29xpq',
    "28e6c",
    "q6gwp",
    "yefrq",
    "pfde9",
]

# Create a local directory to save downloaded files
save_dir = "/Users/tlj258/cslp_data2"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Iterate through each project
for project_id in projects:
    # Specify the project
    project = osf.project(project_id)

    # Access the appropriate storage folder
    storage = project.storage("osfstorage")
    all_data_folder = None
    for item in storage.folders:
        if item.name == f"{project.title} All Data and Scripts":
            all_data_folder = item
            break

    if all_data_folder is None:
        raise FileNotFoundError(
            f"Folder '{project.title} All Data and Scripts' not found in project {project_id}"
        )

    # Iterate through all subject folders within the '{project.title} All Data and Scripts' folder
    for item in all_data_folder.folders:
        if (
            item.name.isdigit()
        ):  # Check if the folder name is an integer (i.e., a subject folder)
            subject_folder = item
            subject_id = item.name

            # Define file names based on the project name
            if project.title == "MMN":
                target_files = [
                    f"{subject_id}_{project.title}_ds_reref_ucbip_hpfilt_ica_prep1.set",
                    f"{subject_id}_{project.title}_ds_reref_ucbip_hpfilt_ica_prep1.fdt",
                    # f"{subject_id}_{project.title}_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
                    # f"{subject_id}_{project.title}_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt"
                ]
            else:
                target_files = [
                    f"{subject_id}_{project.title}_shifted_ds_reref_ucbip_hpfilt_ica_prep1.set",
                    f"{subject_id}_{project.title}_shifted_ds_reref_ucbip_hpfilt_ica_prep1.fdt",
                    # f"{subject_id}_{project.title}_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set",
                    # f"{subject_id}_{project.title}_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.fdt"
                ]

            # Create directory for the project and subject
            project_dir = os.path.join(
                save_dir, f"{project.title} All Data and Scripts", subject_id
            )
            if not os.path.exists(project_dir):
                os.makedirs(project_dir)

            # Iterate through files in the subject folder to find the specific files
            for target_file in target_files:
                print(f"File {target_file}")

                file_found = False
                for file in subject_folder.files:
                    if file.name == target_file:
                        # Specify local file path to save the downloaded file
                        local_file_path = os.path.join(project_dir, file.name)
                        # Download the file
                        with open(local_file_path, "wb") as local_file:
                            file.write_to(local_file)
                        file_found = True
                        break
                if not file_found:
                    print(
                        f"\033[91mFile {target_file} not found in subject folder {subject_id} of project {project.title}\033[0m"
                    )


print("Download complete.")
