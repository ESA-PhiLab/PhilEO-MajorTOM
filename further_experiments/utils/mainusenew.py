                def mean_ioouu(labels, predictions, n_classes, confimage):   
                    mean_iou = 0.0  
                    mean_iou2 = 0.0
                    seen_classes = 0

                    for c in range(n_classes):
                        labels_c = (labels == c)
                        pred_c = (predictions == c)

                        labels_c_sum = (labels_c).sum()
                        pred_c_sum = (pred_c).sum()

                        if (labels_c_sum > 0) or (pred_c_sum > 0): 
                            # # (??)(??)       
                            # # (??)(??)     
                            # # (??)(??) 
                            # # (??)(??) (!) 
                            labels_c = confimage[labellss[vartochange, :, :].detach().cpu().numpy() == c] 
                            # labels_c = (labellss[vartochange, :, :].detach().cpu().numpy() == c)  
                            # pred_c = (theoutput[vartochange, :, :].detach().cpu().numpy() == c)

                            # labels_c_sum = (labels_c).sum()
                            # pred_c_sum = (pred_c).sum()
                            # # (??)(??) (!) 
                            # # (??)(??) 
                            # # (??)(??)    
                            # # (??)(??)

                            
                            
                            
                            
                            seen_classes += 1    

                            intersect = np.nanmean(labels_c)
                            # # this is for each segment - for every segment      
                            # # then, stack together all the mean confidence values  
                            # # next, correlation - pearsonr - the correlation coefficient 
                            # # correlation between the mean confidence values (from here) and the IoU (i.e. forcorrIoU)
                            # # pearsonr
                            # # import pearsonr

                            #intersect = np.logical_and(labels_c, pred_c).sum()
                            #union = labels_c_sum + pred_c_sum - intersect

                            union = 1.
                            
                            
                            
                            #print(intersect / union)                      

                            #print(intersect) 

                            #print(intersect / union)  
                            #print(c)

                            mean_iou += intersect / union  

                            mean_iou2 += intersect 

                    #print(seen_classes)           

                    #return mean_iou / seen_classes if seen_classes else 0 
                    return mean_iou / seen_classes if seen_classes else 0, mean_iou2 / seen_classes if seen_classes else 0 

                