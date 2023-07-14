import glob

if __name__ == "__main__":
    # filepaths = ["../../../CRO_DT_RL/results/DSSAT/simul4/dssat_ppo_p000001_final-model_50simulations_RP_penalizeTrue_4onwards",
    #              "../../../CRO_DT_RL/results/DSSAT/simul4/dssat_ppo_p000001_final-model_50simulations_RP_penalizeTrue_11onwards",
    #              "../../../CRO_DT_RL/results/DSSAT/simul4/dssat_ppo_p000001_final-model_50simulations_RP_penalizeTrue_20onwards",
    #              "../../../CRO_DT_RL/results/DSSAT/simul4/dssat_ppo_p000001_final-model_50simulations_RP_penalizeTrue_25onwards",]
    # output_filepath = "../../../CRO_DT_RL/results/DSSAT/simul4/dssat_ppo_p000001_final-model_50simulations_RP_penalizeTrue"
    # filetype = "RP"

    # filepaths = [
    #     "../../../CRO_DT_RL/results/DSSAT/simul3/dssat_CRO__2023_05_19-21_52_01",
    #     "../../../CRO_DT_RL/results/DSSAT/simul3/dssat_CRO__2023_05_20-21_45_44",
    #     "../../../CRO_DT_RL/results/DSSAT/simul3/dssat_CRO__2023_05_21-07_26_04",
    #     "../../../CRO_DT_RL/results/DSSAT/simul3/dssat_CRO__2023_05_21-23_11_24",
    #     "../../../CRO_DT_RL/results/DSSAT/simul3/dssat_CRO__2023_05_22-17_05_18",
    #     "../../../CRO_DT_RL/results/DSSAT/simul3/dssat_CRO__2023_05_23-09_55_54"
    # ]
    # output_filepath = "../../../CRO_DT_RL/results/DSSAT/simul3/dssat_CRO__2023_05_19-21_52_01"
    # filetype = "CRO_DT_RL"

    # filepaths = [
    #     "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_25-23_02_01",
    #     "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_25-23_02_03",
    #     "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_25-23_03_45",
    #     "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_26-13_53_42",
    #     "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_27-09_09_06",
    #     "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_27-17_55_43",
    #     "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_27-22_43_09",
    #     "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_27-22_43_18",
    #     "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_28-12_49_28",
    #     "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_28-15_57_58",
    # ]
    # output_filepath = "../../../CRO_DT_RL/results/DSSAT/dssat_CRO-IL-RP__2023_05_25-23_02_01"
    # filetype = "CRO_DT_RL"

    filepaths = [
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP__2023_05_26-13_53_42",
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP_start9_2023_05_29-23_57_24",
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP_start9_2023_05_30-13_23_57",
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP_start9_2023_06_01-19_28_27",
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP__2023_05_31-15_52_15",
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP__2023_06_01-14_28_47",
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP__2023_06_01-23_17_15",
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP__2023_05_29-16_40_00",
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP__2023_05_30-20_43_10",
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP__2023_06_01-15_15_25",
        "../../../CRO_DT_RL/results/DSSAT/cauldron/dssat_CRO-IL-RP__2023_06_02-08_14_55"
    ]
    output_filepath = "../../../CRO_DT_RL/results/DSSAT/cauldron/complete_1"
    filetype = "CRO_DT_RL"

    file_headers = []
    file_contents = []
    file_tmp_contents = []

    for i, filepath_prefix in enumerate(filepaths):
        filepath = filepath_prefix + ".txt"
        filepath_tmp = filepath_prefix + "_tmp.txt"

        with open(filepath, "r") as f:
            lines = f.readlines()
            for j, line in enumerate(lines):
                if line.startswith("Tree #"):
                    lines[0] = filepath + "\n\n" + lines[0]
                    file_headers.append(lines[:j])
                    file_contents.append(lines[j:])
                    break

        if filetype == "CRO_DT_RL":
            with open(filepath_tmp, "r") as f:
                lines = f.readlines()
                for j, line in enumerate(lines):
                    if line.startswith("Generation #"):
                        file_tmp_contents.append(lines[j:])
                        break
        else:
            with open(filepath_tmp, "r") as f:
                lines = f.readlines()
                file_tmp_contents.append(lines)

    total_trees = 0
    for fc in file_contents:
        total_trees += len([l for l in fc if l.startswith("Tree #")])

    print(f"Saving to {output_filepath + '_condensed.txt'}")
    with open(output_filepath + "_condensed.txt", "w") as file:
        output = ""
        output += f"Total trees: {total_trees}\n\n"
        output += "".join([f for file in file_headers for f in file])
        output += "".join([f for file in file_contents for f in file])
        file.writelines(output)

    print(f"Saving to {output_filepath + '_condensed_tmp.txt'}")
    with open(output_filepath + "_condensed_tmp.txt", "w") as file:
        output = ""
        output += "".join([f for file in file_headers for f in file])
        output += "".join([f for file in file_tmp_contents for f in file])
        file.writelines(output)