import os
import copy
from collections import deque
import numpy as np
import cv2
import torch
import torch.optim as optim
from tqdm import tqdm
from . import softtriple

def train_softtriplet(data_streams, writer, max_steps, n_class, lr,
                      model_path='model', model_save_interval=2000,
                      tsne_test_interval=1000, n_test_data=1000, pretrained=None,
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    stream_train, stream_train_eval, stream_test = data_streams
    epoch_iterator = stream_train.get_epoch_iterator()
    test_data = deque(maxlen=n_test_data)
    test_label = deque(maxlen=n_test_data)
    test_img = deque(maxlen=n_test_data)

    stri = softtriple.SoftTripleNet(n_class=n_class, pretrained=pretrained).to(device)
    optimizer = optim.Adam(stri.parameters(), lr=lr, weight_decay=1.0e-4)

    img_mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 3, 1, 1)
    cnt = 0

    with tqdm(total=max_steps) as pbar:
        for batch in copy.copy(epoch_iterator):
            x_batch, label = batch
            x_batch -= img_mean
            pbar.update(1)

            loss, embedding = stri(torch.from_numpy(x_batch).to(device),
                                   torch.from_numpy(label.astype(np.int64)).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description("Loss: %f" % loss.item())

            if cnt > 0 and cnt % model_save_interval == 0:
                torch.save(stri.state_dict(), os.path.join(model_path, 'model_%d.pth' % cnt))

            if cnt > 0 and n_test_data > 0 and cnt % tsne_test_interval == 0:
                writer.add_embedding(np.vstack(test_data), np.vstack(test_label).flatten(),
                                     torch.from_numpy(np.stack(test_img, axis=0)),
                                     global_step=cnt, tag='embedding/train')
                writer.flush()

            writer.add_scalar('Loss', loss.item(), cnt)

            test_data.extend(embedding.detach().cpu().numpy())
            test_label.extend(label)
            for x in x_batch:
                xx = cv2.resize(x.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
                test_img.append((xx + img_mean[0]) / 255.0)
            cnt += 1
            if cnt > max_steps:
                break
