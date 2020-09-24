import solaris as sol
import sys
import os
config_path = 'yml/sn7_baseline_train.yml'
config = sol.utils.config.parse(config_path)
config['training_data_csv'] = os.path.abspath('./output/csvs/sn7_baseline_train_df.csv')
config['training']['callbacks']['model_checkpoint']['filepath'] = os.path.abspath('./models/sn7_baseline/xdxd_best.pth')
config['training']['model_dest_path'] = os.path.abspath('./models/sn7_baseline/xdxd_final.pth')
print('Config:')
print(config)

# make model output dir
os.makedirs(os.path.dirname(config['training']['model_dest_path']), exist_ok=True)

trainer = sol.nets.train.Trainer(config=config)
trainer.train()
