## CONTRASTE: Supervised Contrastive Pre-training With Aspect-based Prompts For Aspect Sentiment Triplet Extraction

EMNLP 2023 Anonymous Submission For Long Paper ID: 5841

## Supervised Contrastive Learning (SCL)-based Pre-Training

To pretrain the model and save the chekpoints of the pretrained models after certain epochs use:

```
python pretrain.py --pretrain_datapath Contrastive/TemplateFineGrainedContrast.json \
                   --model_name_or_path t5-base\
                   --batch_size 16 \
                   --epochs 20 \
     
 ```

We will release the pre-training code and pre-processed dataset upon acceptance of the work.


## ASTE Fine-tuning Without Pre-Training

To finetune for ASTE task without pretraining:
 
 ```
!python main.py --task 15res \
                --train_dataset_path 15res/train \
                --dev_dataset_path 15res/dev \
                --test_dataset_path 15res/test \
                --model_name_or_path t5-base \
                --n_gpu 1 \
                --do_train \
                --do_eval \
                --train_batch_size 2 \
                --gradient_accumulation_steps 2 \
                --eval_batch_size 16 \
                --learning_rate 3e-4 \
                --num_train_epochs 20 \
                --regressor True \
                --use_tagger True \
                --beta 0.2 \
                --alpha 0.8 \
                --logger_name 15res_regressor_tagger_base.txt \
                --log_message regressor_and_tagger_0 \
 
 ```

 
## ASTE Fine-tuning From SCL-Pre-Trained Checkpoint

To finetune the pretrained model on the ASTE Task using a particular checkpoint:
 
 ```
!python main.py --task 15res \
                --train_dataset_path 15res/train \
                --dev_dataset_path 15res/dev \
                --test_dataset_path 15res/test \
                --model_name_or_path models/contraste_model_after_14_epochs\
                --n_gpu 1 \
                --do_train \
                --do_eval \
                --train_batch_size 2 \
                --gradient_accumulation_steps 2 \
                --eval_batch_size 16 \
                --learning_rate 3e-4 \
                --num_train_epochs 20 \
                --regressor True \
                --use_tagger True \
                --beta 0.2 \
                --alpha 0.8 \
                --model_weights models/contraste_model_after_2_epochs \
                --logger_name 15res_logs_regressor_tagger_contrast2.txt \
                --log_message regressor_and_tagger_2 \
     
 ```
 
### Packages Required
  
  * datasets
  * pytorch_lightning
  * sentencepiece
  * transformers

