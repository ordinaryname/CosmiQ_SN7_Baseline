import solaris as sol
import os
config_path = 'yml/sn7_baseline_infer.yml'
config = sol.utils.config.parse(config_path)
config['model_path'] = os.path.abspath('./models/sn7_baseline/xdxd_final.pth')
config['inference_data_csv'] = os.path.abspath('./output/csvs/sn7_baseline_test_df.csv') 
config['inference']['output_dir'] = os.path.abspath('./inference_out/sn7_baseline_preds/raw')
print('Config:')
print(config)

# make infernce output dir
os.makedirs(os.path.dirname(config['inference']['output_dir']), exist_ok=True)

inferer = sol.nets.infer.Inferer(config)
inferer()
