import gpt_2_simple as gpt2

# gpt2.download_gpt2(model_name='124M')


file_name = 'written_test.txt'

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=100,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=50,
              save_every=20
              )

gpt2.generate(sess, run_name='run1')
gpt2.generate(sess, length=30,temperature=0.7, # 낮을 수록 자유도 떨어져서 이미 있는 텍스트 반복할 가능성 높아짐
              prefix='옛날에',
              nsamples=5, batch_size=5, top_k=40)
