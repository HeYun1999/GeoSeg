def on_validation_epoch_end(self):
    # Intersection_over_Union()，获得每一类的iou
    if 'vaihingen' in self.config.log_name:
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
        F1 = np.nanmean(self.metrics_val.F1()[:-1])
        Acc = np.nanmean(self.metrics_val.Pixel_Accuracy_Class()[:-1])
        Recall = np.nanmean(self.metrics_val.Recall()[:-1])

        acc_per_class = self.metrics_val.Pixel_Accuracy_Class()[:-1]
        iou_per_class = self.metrics_val.Intersection_over_Union()[:-1]
        f1_per_class = self.metrics_val.F1()[:-1]
        recall_per_class = self.metrics_val.Recall()[:-1]

    elif 'potsdam' in self.config.log_name:
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
        F1 = np.nanmean(self.metrics_val.F1()[:-1])
    elif 'whubuilding' in self.config.log_name:
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
        F1 = np.nanmean(self.metrics_val.F1()[:-1])
    elif 'massbuilding' in self.config.log_name:
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
        F1 = np.nanmean(self.metrics_val.F1()[:-1])
    elif 'cropland' in self.config.log_name:
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
        F1 = np.nanmean(self.metrics_val.F1()[:-1])
    else:
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_val.F1())
        Acc = np.nanmean(self.metrics_val.Pixel_Accuracy_Class())
        Recall = np.nanmean(self.metrics_val.Recall())

        acc_per_class = self.metrics_val.Pixel_Accuracy_Class()
        iou_per_class = self.metrics_val.Intersection_over_Union()
        f1_per_class = self.metrics_val.F1()
        recall_per_class = self.metrics_val.Recall()

    OA = np.nanmean(self.metrics_val.OA())

    eval_value = {'mAcc': Acc,
                  'mIoU': mIoU,
                  'mF1': F1,
                  'mRecall': Recall,
                  # 'mOA': OA
                  }
    print('val:', eval_value)

    acc_value = {}
    iou_value = {}
    f1_value = {}
    recall_value = {}
    # acc
    for class_name, acc in zip(self.config.classes, acc_per_class):
        acc_value[class_name] = acc
    print("acc_value:")
    print(acc_value)
    # iou
    for class_name, iou in zip(self.config.classes, iou_per_class):
        iou_value[class_name] = iou
    print("iou_value:")
    print(iou_value)
    # f1
    for class_name, f1 in zip(self.config.classes, f1_per_class):
        f1_value[class_name] = f1
    print("f1_value:")
    print(f1_value)
    # recall
    for class_name, recall in zip(self.config.classes, recall_per_class):
        recall_value[class_name] = recall
    print("recall_value:")
    print(recall_value)

    self.metrics_val.reset()
    # log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
    # self.log_dict(log_dict, prog_bar=True)

    # self.log_dict(eval_value, prog_bar=True)

    # self.log_dict(acc_value, prog_bar=False)

    # self.log_dict(iou_value, prog_bar=False)

    # self.log_dict(f1_value, prog_bar=False)

    # self.log_dict(recall_value, prog_bar=False)