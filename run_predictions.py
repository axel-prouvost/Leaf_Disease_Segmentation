from EasIlastik.run_ilastik import run_ilastik
run_ilastik(
    input_path="C:/Users/Axel/Documents/Mission_RD/Feuilles",
    model_path="C:/Users/Axel/Documents/Mission_RD/models/leaf_5.ilp",
    result_base_path="C:/Users/Axel/Documents/Mission_RD/h5_v2/normal_5_h5/",
    export_source="Probabilities",
    output_format="hdf5"
)