
# 蒋老师要求的test
def test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test):
    criterion = torch.nn.CrossEntropyLoss()
    n_classes = 6
    _loss = 0
    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]

    # 用于收集所有样本的预测概率和实际标签
    all_preds = []
    all_labels = []

    # 用于存储结果以写入 Excel
    results = []

    print('Start Testing')
    pbar = tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.eval()

    for step, (spec, image, label) in enumerate(test_dataloader):
        with torch.no_grad():
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            out_mm, out_m1, out_m2, a, v = net(spec.unsqueeze(1).float(), image.float())

            loss_m1 = criterion(out_m1, label)
            loss_m2 = criterion(out_m2, label)
            loss = loss_m1 + loss_m2
            _loss += loss.item()

            # 初始化输出张量
            out = torch.zeros_like(out_m1)

            for i in range(len(label)):
                preds_m1 = torch.max(out_m1[i], 0)[1]  # 模态1预测结果
                preds_m2 = torch.max(out_m2[i], 0)[1]  # 模态2预测结果

                if preds_m1 == preds_m2:
                    out[i] = (out_m1[i] + out_m2[i]) / 2
                else:
                    prob_m1 = torch.nn.functional.softmax(out_m1[i], dim=0)  # 模态1的概率
                    prob_m2 = torch.nn.functional.softmax(out_m2[i], dim=0)  # 模态2的概率
                    conf_m1 = prob_m1[preds_m1]  # 模态1预测类的概率
                    conf_m2 = prob_m2[preds_m2]  # 模态2预测类的概率
                    out[i] = out_m1[i] if conf_m1 >= conf_m2 else out_m2[i]

            probs = torch.nn.functional.softmax(out, dim=1)  # 计算最终的概率
            preds = torch.max(probs, 1)[1]  # 获得最终预测结果

            # 更新准确率统计
            correct = (preds == label).float()
            for i in range(len(label)):
                num[label[i].item()] += 1
                acc[label[i].item()] += correct[i].item()

                # 获取对目标类的预测概率
                target_class_prob = probs[i, label[i]].item()
                is_correct = preds[i].item() == label[i].item()

                # 将结果存入列表
                results.append([label[i].item(), target_class_prob, is_correct])

            all_preds.append(probs.cpu().numpy())
            all_labels.append(label.cpu().numpy())

            pbar.set_postfix(**{'test_loss': _loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    pbar.close()

    # 将预测概率和标签从列表转换为NumPy数组
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算mAP
    mAP = 0.0
    for i in range(n_classes):
        label_binary = (all_labels == i).astype(int)
        mAP += average_precision_score(label_binary, all_preds[:, i])
    mAP /= n_classes

    # 计算总体精度
    accuracy = sum(acc) / sum(num)

    # 将结果写入Excel文件
    df = pd.DataFrame(results, columns=['Label', 'Target Class Probability', 'Is Correct'])
    df.to_excel('test_results_.xlsx', index=False)

    return _loss / epoch_step_test, accuracy, mAP

# 原始train和test
def train(args, epoch, net, device, train_dataloader, optimizer, scheduler, epoch_step_train):

    criterion = nn.CrossEntropyLoss()
    _total_loss = 0

    print('Start Training')
    pbar = tqdm(total=epoch_step_train, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.train()

    for step, bag in enumerate(train_dataloader):
        spec = bag[0].to(device)
        image = bag[1].to(device)
        label = bag[2].to(device)

        optimizer.zero_grad()
        out_mm, out_m1, out_m2, a, v = net(spec.unsqueeze(1).float(), image.float())

        loss1 = criterion(out_mm, label)
        loss2 = criterion(out_m1, label)
        loss3 = criterion(out_m2, label)
        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()

        _total_loss += loss.item()

        pbar.set_postfix(**{'train_loss': _total_loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
        pbar.update(1)

    pbar.close()
    scheduler.step() # 学习率更新

    return _total_loss / epoch_step_train

def test(args, epoch, net, device, test_dataloader, optimizer, epoch_step_test):
    criterion = torch.nn.CrossEntropyLoss()
    n_classes = 6
    _loss = 0
    num = [0.0 for _ in range(n_classes)]
    acc = [0.0 for _ in range(n_classes)]

    # 用于收集所有样本的预测概率和实际标签
    all_preds = []
    all_labels = []

    print('Start Testing')
    pbar = tqdm(total=epoch_step_test, desc=f'Epoch {epoch + 1}/{args.epochs}', postfix=dict, mininterval=0.3, ncols=120)
    net.eval()

    for step, (spec, image, label) in enumerate(test_dataloader):
        with torch.no_grad():
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            # out_mm, a, v = net(spec.unsqueeze(1).float(), image.float())
            out_mm, out_m1, out_m2, a, v = net(spec.unsqueeze(1).float(), image.float())

            loss_mm = criterion(out_mm, label)
            loss_m1 = criterion(out_m1, label)
            loss_m2 = criterion(out_m2, label)
            loss = loss_mm + loss_m1 + loss_m2
            _loss += loss.item()

            # Softmax函数计算概率
            out = out_m2+out_m1+out_mm
            probs = torch.nn.functional.softmax(out, dim=1)
            preds = torch.max(probs, 1)[1]  # 获得预测结果

            # 更新准确率统计
            correct = (preds == label).float()
            for i in range(len(label)):
                num[label[i].item()] += 1
                acc[label[i].item()] += correct[i].item()

            all_preds.append(probs.cpu().numpy())
            all_labels.append(label.cpu().numpy())

            pbar.set_postfix(**{'test_loss': _loss / (step + 1), 'lr': optimizer.param_groups[0]['lr']})
            pbar.update(1)

    pbar.close()

    # 将预测概率和标签从列表转换为NumPy数组
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算mAP
    mAP = 0.0
    for i in range(n_classes):
        # 注意，average_precision_score需要真实标签为二进制形式（例如one-hot编码）
        label_binary = (all_labels == i).astype(int)
        mAP += average_precision_score(label_binary, all_preds[:, i])
    mAP /= n_classes

    # 计算总体精度
    accuracy = sum(acc) / sum(num)

    return _loss / epoch_step_test, accuracy, mAP