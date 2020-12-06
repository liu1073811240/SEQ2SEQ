import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import Sampling_train_num
from utils import LabeltoStr, StrtoLabel
from utils2 import tensor_to_PIL


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True)

    def forward(self, x):
        # print(np.shape(x))  # torch.Size([64, 3, 60, 240])
        x = x.reshape(-1, 180, 240).permute(0, 2, 1)
        # print(np.shape(x))  # torch.Size([64, 240, 180])

        x = x.reshape(-1, 180)
        # print(np.shape(x))  # torch.Size([15360, 180])

        fc1 = self.fc1(x)
        # print(np.shape(fc1))  # torch.Size([15360, 128])

        fc1 = fc1.reshape(-1, 240, 128)
        # print(np.shape(fc1))  # torch.Size([64, 240, 128])

        lstm, (h_n, h_c) = self.lstm(fc1, None)
        # print(np.shape(lstm))  # torch.Size([64, 24440, 128])

        out = lstm[:, -1, :]
        # print(np.shape(out))  # torch.Size([64, 128])

        return out


class Decoder(nn.Module):
    def __init__(self, bidirectional=True):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True)
        self.out = nn.Linear(128, 62)
        # if bidirectional == True:
        #     self.out = nn.Linear(128*2, 62)
        # else:
        #     self.out = nn.Linear(128, 62)  # 定义全连接层

    def forward(self, x):
        # print(np.shape(x))  # torch.Size([64, 128])

        x = x.reshape(-1, 1, 128)
        # print(np.shape(x))  # torch.Size([64, 1, 128])

        x = x.expand(-1, 4, 128)
        # print(np.shape(x))  # torch.Size([64, 4, 128])

        lstm, (h_n, h_c) = self.lstm(x, None)
        # print(np.shape(lstm))  # torch.Size([64, 4, 128])

        y1 = lstm.reshape(-1, 128)
        # print(np.shape(y1))  # torch.Size([256, 128])

        out = self.out(y1)
        # print(np.shape(out))  # torch.Size([256, 62])

        output = out.reshape(-1, 4, 62)  # 10表示输出十个值，可以更改
        # print(np.shape(output))  # torch.Size([64, 4, 10])  62

        return output


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)

        return decoder


if __name__ == '__main__':
    BATCH = 512
    EPOCH = 100
    save_path = r'params/seq2seq.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MainNet().to(device)

    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("No Params!")

    train_data = Sampling_train_num.Sampling(root="./code")
    valida_data = Sampling_train_num.Sampling(root="./code2")
    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=BATCH, shuffle=True, num_workers=4)
    valida_loader = data.DataLoader(dataset=valida_data, batch_size=64, shuffle=True, num_workers=4)

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            # print(np.shape(x))  # torch.Size([64, 3, 60, 240])
            # print(np.shape(y))  # torch.Size([64, 4, 62])

            batch_x = x.to(device)
            batch_y = y.float().to(device)

            output = net(batch_x)
            # print(np.shape(output))  # torch.Size([64, 4, 62])

            loss = loss_func(output, batch_y)
            # print(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 4 == 0:
                label_y = torch.argmax(y, 2).detach().numpy()
                # print(label_y)

                out_y = torch.argmax(output, 2).cpu().detach().numpy()
                # print(out_y)
                # print(np.sum(out_y == label_y, dtype=np.float32))

                accuracy = np.sum(
                    out_y == label_y, dtype=np.float32) / (BATCH * 4)
                print("epoch:{},i:{},loss:{:.4f},acc:{:.2f}%"
                      .format(epoch, i, loss.item(), accuracy * 100))

                # print("label_y:", LabeltoStr(label_y[0]))
                # print("out_y:", LabeltoStr(out_y[0]))

        torch.save(net.state_dict(), save_path)

        # 训练的时候效果很好，测试时效果就会很差。因为会有不同背景的干扰，且训练数据不多。
        # 训练时所用的背景和测试时所用的背景最好一样、或者用opencv先提取字符轮廓、或者先框中某个字符再进行识别。

        for i, (x, y) in enumerate(valida_loader):
            # print(np.shape(x))  # torch.Size([64, 3, 60, 240])
            # print(np.shape(y))  # torch.Size([64, 4, 62])

            batch_x = x.to(device)
            batch_y = y.float().to(device)

            output = net(batch_x)
            # print(np.shape(output))  # torch.Size([64, 4, 62])

            loss = loss_func(output, batch_y)
            # print(loss.item())

            if i % 4 == 0:
                label_y = torch.argmax(y, 2).detach().numpy()
                # print(label_y)

                out_y = torch.argmax(output, 2).cpu().detach().numpy()
                # print(out_y)
                # print(np.sum(out_y == label_y, dtype=np.float32))

                accuracy = np.sum(
                    out_y == label_y, dtype=np.float32) / (BATCH * 4)
                print("valida_epoch:{},i:{},loss:{:.4f},acc:{:.2f}%"
                      .format(epoch, i, loss.item(), accuracy * 100))

                print("label_y:", LabeltoStr(label_y[0]))
                print("out_y:", LabeltoStr(out_y[0]))
