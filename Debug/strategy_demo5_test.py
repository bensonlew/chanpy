import json
from typing import Dict, TypedDict
import sys

import xgboost as xgb

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver
import gc
from memory_profiler import profile

# TODO: feature 选择5分钟一段，  30 分钟的级别买点， 此时是否为确认的5分钟



class T_SAMPLE_INFO(TypedDict):
    feature: CFeatures
    is_buy: bool
    open_time: CTime


def plot(chan, plot_marker):
    plot_config = {
        "plot_kline": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_segseg": True,
        "plot_segbsp": True,
        "plot_segzs": True,
        "plot_macd": True,
        "plot_zs": True,
        "plot_bsp": True,
        "plot_marker": True,
    }
    plot_para = {
        "figure": {
            "x_range": 30000,
            "w": 360,
            "h": 24,
            "x_tick_num": 100
        },
        "marker": {
            "markers": plot_marker
        }
    }
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.save2img("label.png")

# @profile
def plot_sub(chan, plot_marker=None, date=None):
    plot_config = {
        "plot_kline": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_segseg": True,
        "plot_segbsp": True,
        "plot_segzs": True,
        "plot_macd": True,
        "plot_zs": True,
        "plot_bsp": True,
        "plot_marker": True,
    }
    plot_para = {
        "figure": {
            "x_range": 3000,
            "w": 60,
            "h": 24,
            "x_tick_num": 10
        },
        "marker": {
            "markers": plot_marker
        }
    }
    plot_driver = CPlotDriver(
        chan,
        plot_config=plot_config,
        plot_para=plot_para,
    )
    plot_driver.save2img("{}.png".format(date))
    plot_driver = None
    gc.collect()


def stragety_feature(last_klu):
    return {
        "open_klu_rate": (last_klu.close - last_klu.open)/last_klu.open,
    }


if __name__ == "__main__":
    """
    本demo主要演示如何记录策略产出的买卖点的特征
    然后将这些特征作为样本，训练一个模型(以XGB为demo)
    用于预测买卖点的准确性

    请注意，demo训练预测都用的是同一份数据，这是不合理的，仅仅是为了演示
    """
    # code = "000002"
    code = sys.argv[1]
    # begin_time = "2021-04-01"
    # end_time = "2022-04-01"
    begin_time = None
    end_time = None 
    data_src = "custom:parquetAPI.Parquet_API"
    lv_list = [KL_TYPE.K_5M]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "bi_strict": False,
        "bi_allow_sub_peak": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    bsp_dict: Dict[int, T_SAMPLE_INFO] = {}  # 存储策略产出的bsp的特征

    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        cur_lv_chan = chan_snapshot[0] #当前kline对象
        # if last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-2].idx == last_bsp.klu.klc.idx:
        if last_bsp.is_buy and last_bsp.klu.idx not in bsp_dict and cur_lv_chan[-1].lst[-1].close > last_bsp.klu.klc.high:
            # 假如策略是：买卖点分形第三元素出现时交易
            bsp_dict[last_bsp.klu.idx] = {
                "idx": last_klu.idx,
                "close": last_klu.close,
                "feature": last_bsp.features,
                "is_buy": last_bsp.is_buy,
                "open_time": last_klu.time                
            }
            if len(cur_lv_chan.seg_list) > 0:
                now_seg = {
                    "begin": cur_lv_chan.seg_list[-1].start_bi.begin_klc.lst[0].idx, 
                    "end": cur_lv_chan.seg_list[-1].end_bi.end_klc.lst[-1].idx,
                    "high": cur_lv_chan.seg_list[-1]._high(),
                    "low": cur_lv_chan.seg_list[-1]._low(),
                    "is_sure": cur_lv_chan.seg_list[-1].is_sure,
                    "bi_num": len(cur_lv_chan.seg_list[-1].bi_list)
                }
                bsp_dict[last_bsp.klu.idx]["feature"].add_feat({
                   "now_seg": now_seg
                })

                 
            # print("now_seg", now_seg)
            if len(cur_lv_chan.seg_list) > 1:
                last_seg = {
                    "begin": cur_lv_chan.seg_list[-2].start_bi.begin_klc.lst[0].idx,
                    "end": cur_lv_chan.seg_list[-2].end_bi.end_klc.lst[0].idx,
                    "high": cur_lv_chan.seg_list[-2]._high(),
                    "low": cur_lv_chan.seg_list[-2]._low(),
                    "is_sure": cur_lv_chan.seg_list[-2].is_sure,
                    "bi_num": len(cur_lv_chan.seg_list[-2].bi_list)
                }
                bsp_dict[last_bsp.klu.idx]["feature"].add_feat({
                    "last_seg": last_seg
                })
            
            now_segzs = []
            last_segzs = []

            for zs in cur_lv_chan.zs_list.zs_lst:
                zs_begin = zs.begin.idx
                zs_end = zs.end.idx
                # print(zs_begin, zs_end)
                if len(cur_lv_chan.seg_list) > 0:
                    if zs_end > now_seg["begin"]:
                        now_segzs.append({
                            "begin": zs_begin,
                            "end": zs_end
                        })
                        

                if len(cur_lv_chan.seg_list) > 1:
                    if zs_begin < last_seg["end"] and zs_end > last_seg["begin"]:
                        last_segzs.append({
                            "begin": zs_begin,
                            "end": zs_end
                        })
            
            bsp_dict[last_bsp.klu.idx]["feature"].add_feat({
                "now_segzs": now_segzs,
                "last_segzs": last_segzs
            })

            # 所在段，上一段， 段中枢
            if len(cur_lv_chan.segseg_list) > 0:
                now_segseg = {
                    "begin": cur_lv_chan.segseg_list.lst[-1].bi_list[0].start_bi.begin_klc.lst[0].idx, 
                    "end": cur_lv_chan.segseg_list.lst[-1].bi_list[-1].end_bi.end_klc.lst[-1].idx,
                    "high": cur_lv_chan.segseg_list[-1]._high(),
                    "low": cur_lv_chan.segseg_list[-1]._low(),
                    "is_sure": cur_lv_chan.segseg_list[-1].is_sure,
                    "bi_num": len(cur_lv_chan.segseg_list[-1].bi_list)
                }
                bsp_dict[last_bsp.klu.idx]["feature"].add_feat({
                   "now_segseg": now_segseg
                }) 


            if len(cur_lv_chan.segseg_list) > 1:
                last_segseg = {
                    "begin": cur_lv_chan.segseg_list[-2].bi_list[0].start_bi.begin_klc.lst[0].idx,
                    "end": cur_lv_chan.segseg_list[-2].bi_list[-1].end_bi.end_klc.lst[-1].idx,
                    "high": cur_lv_chan.segseg_list[-2]._high(),
                    "low": cur_lv_chan.segseg_list[-2]._low(),
                    "is_sure": cur_lv_chan.segseg_list[-2].is_sure,
                    "bi_num": len(cur_lv_chan.segseg_list[-2].bi_list)
                }
                bsp_dict[last_bsp.klu.idx]["feature"].add_feat({
                    "last_segseg": last_segseg
                })
            
            now_segzs = []
            last_segzs = []

            for segzs in cur_lv_chan.segzs_list:
                segzs_begin = segzs.begin.idx
                segzs_end = segzs.end.idx
                if len(cur_lv_chan.segzs_list) > 0:
                    if segzs_end > now_seg["begin"]:
                        now_segzs.append({
                            "begin": segzs_begin,
                            "end": segzs_end
                        })
                        

                if len(cur_lv_chan.segseg_list) > 1:
                    if segzs_begin < last_segseg["end"] and segzs_end > last_segseg["begin"]:
                        last_segzs.append({
                            "begin": segzs_begin,
                            "end": segzs_end
                        })
            
            bsp_dict[last_bsp.klu.idx]["feature"].add_feat({
                "now_seg_segzs": now_segzs,
                "last_seg_segzs": now_segzs
            })
                

            bsp_dict[last_bsp.klu.idx]['feature'].add_feat(stragety_feature(last_klu))  # 开仓K线特征
            print(last_bsp.klu.time, last_bsp.is_buy)
            sub_plot_marker = {}
            for bsp_klu_idx, feature_info in bsp_dict.items():
                bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]
                label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label

                # sub_plot_marker[feature_info["open_time"].to_str()] = ("√" if label else "×", "down" if feature_info["is_buy"] else "up")
            # plot_sub(chan, plot_marker=sub_plot_marker, date=last_bsp.klu.time.to_str().replace(" ", "_").replace("/", "-"))

    # 生成libsvm样本特征
    bsp_academy = [bsp.klu.idx for bsp in chan.get_bsp()]

    # 一买的前一段，一卖的后一段段作为买入成功的标志
    buy_range1 = []
    buy_range2 = []
    cur_lv_chan = chan[0]
    for segseg in cur_lv_chan.segseg_list:
        start = segseg.bi_list[0].start_bi.begin_klc.lst[0].idx
        end = segseg.bi_list[-1].end_bi.end_klc.lst[-1].idx
        start_last_seg = segseg.bi_list[-1].start_bi.begin_klc.lst[0].idx
        if segseg.is_sure:
            if segseg.is_up():
                low = segseg._low()
                high = segseg._high()
                buy_range1.append([start, end, low, high])
                if len(buy_range2) > 0:
                    buy_range2[-1].extend([low, high])
            elif segseg.is_down():
                buy_range2.append([start_last_seg, end])

    print(buy_range1)
    print(buy_range2)


    def get_label(feature_info, buy_range1, buy_range2):
        idx = feature_info["idx"]
        close = feature_info["close"]
        for buy1 in buy_range1:     
                   
            if idx >= buy1[0] and idx <= buy1[1]:
                if len(buy1) < 3:
                    return "1unknown"
                if (buy1[3] - close) > 2 * (close - buy1[2]):
                    return "1low"
                else:
                    return "1high"
            
        for buy2 in buy_range2:
            if idx >= buy2[0] and idx <= buy2[1]:
                if len(buy2) < 3:
                    return "2unknown"
                if (buy2[3] - close) > 2 * (close - buy2[2]):
                    return "2low"
                else:
                    return "2high"
        return 0

    # seg_bsp_academy = [seg_bsp.klu.idx for seg_bsp in chan.get_seg_bsp()]
    feature_meta = {}  # 特征meta
    cur_feature_idx = 0
    plot_marker = {}
    fid = open("{}.feature.libsvm.json".format(code), "w")
    for bsp_klu_idx, feature_info in bsp_dict.items():
        # label = int(bsp_klu_idx in bsp_academy)  # 以买卖点识别是否准确为label
        label = get_label(feature_info, buy_range1, buy_range2)
        features = []  # List[(idx, value)]
        # for feature_name, value in feature_info['feature'].items():
        #     if feature_name not in feature_meta:
        #         feature_meta[feature_name] = cur_feature_idx
        #         cur_feature_idx += 1
        #     features.append((feature_meta[feature_name], value))
        # features.sort(key=lambda x: x[0])
        # feature_str = " ".join([f"{idx}:{value}" for idx, value in features])
        # print(feature_info['feature'].features())
        feature_str = json.dumps(feature_info['feature'].features())
        time = feature_info["open_time"].to_str()
        fid.write(f"{time}\t{label}\t{feature_str}\n")
        plot_marker[feature_info["open_time"].to_str()] = ("√" if label else "×", "down" if feature_info["is_buy"] else "up")
    fid.close()

    with open("feature.meta", "w") as fid:
        # meta保存下来，实盘预测时特征对齐用
        fid.write(json.dumps(feature_meta))

    # 画图检查label是否正确
    # plot(chan, plot_marker)

    # # load sample file & train model
    # dtrain = xgb.DMatrix("feature.libsvm?format=libsvm")  # load sample
    # param = {'max_depth': 2, 'eta': 0.3, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
    # evals_result = {}
    # bst = xgb.train(
    #     param,
    #     dtrain=dtrain,
    #     num_boost_round=10,
    #     evals=[(dtrain, "train")],
    #     evals_result=evals_result,
    #     verbose_eval=True,
    # )
    # bst.save_model("model.json")

    # # load model
    # model = xgb.Booster()
    # model.load_model("model.json")
    # # predict
    # print(model.predict(dtrain))
