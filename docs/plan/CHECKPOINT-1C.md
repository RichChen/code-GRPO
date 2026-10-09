# CHECKPOINT-1C（提前停下）：第 5b 步近似重复检查超过 10%

## 结论

- 按审计修订 2 第 5 条做近似重复检查：留出集 **321 个来源函数里有 63 个（19.6%）** 和训练集函数的最大 Jaccard 超过 0.8，超过了"移动的超过留出集函数总数的 10% 就停下报告"的门槛。
- 所以停在第 5b 步：**没有移动任何函数，没有重建题集，没有开始第 6 步和第 7 步**。没有自行改阈值、改规范化方式或改划分方法。
- 抽查结果显示这些大多是真重复，不是规范化过头造成的误报（例子见下）。其中不少对只差一个运算符或常量，比如 `remove_even` 和 `remove_odd` 只差 `!=` 和 `==`，而这正是一道 `compare_swap` 题的标准答案。

## 已完成的部分

| 项 | 状态 |
| --- | --- |
| "审计修订 2"原样补进计划第 5 步末尾，单独提交 | 完成（`8dbbf1b`） |
| `nvidia-smi` 前置检查 | 能跑通：RTX 2080 Ti，11264 MiB 显存，驱动 615.78.05，支持 CUDA 13.4。第 6 步的前置条件满足，但还没装 vLLM |
| 下载完整版 MBPP，校验 sha256，写进 `SOURCES.md` | 完成：`mbpp.jsonl`，同一仓库同一 commit（`f82046b`），sha256 `ccf64cea…a92a9f`，Apache-2.0 |
| `sources.py` 合并 sanitized 版和完整版，加入 `challenge_test_list` | 完成，有测试 |
| 隐藏用例加强 `taskgen/augment.py` | 代码和测试完成，在 60 个函数上试跑过；**还没接进 `build.py`** |
| 近似重复检查 `taskgen/dedup.py` | 代码和测试完成；在 seed 0 的划分上运行后触发停止条件 |
| 接进构建、重建题集、两次构建对比 sha256、50 题脚本化检查 | 未做（停在近似重复检查） |
| 第 6 步、第 7 步 | 未做 |

全量测试 181 passed（新增 `tests/test_augment_dedup.py` 12 个）。

### 来源函数（合并完整版 MBPP 之后）

| 来源 | 原始题数 | 保留 | 丢弃原因 |
| --- | --- | --- | --- |
| MBPP sanitized（沿用 sanitized 版的代码和断言） | 427 | 393 | 合格断言不足 3 条 34 |
| MBPP 完整版新增（不在 sanitized 里的题号） | 547 | 530 | 合格断言不足 3 条 15，`test_setup_code` 非空 2 |
| HumanEval | 164 | 147 | 合格断言不足 3 条 17 |
| 合计 | 1138 | **1070** | |

- 完整版新增部分的断言 1635 条，保留 1601 条。丢弃原因：容器里有浮点数 16、参数不是字面量 9、重复调用 6、调用被包装 3。
- `challenge_test_list` 只有 11 题有，拆出 16 条，全部合格（sanitized 题号 6 条，完整版新增 10 条），作为 `kind="challenge"` 只进隐藏用例。原始代码跑不过的 challenge 用例在加强阶段丢掉并计数。
- seed 0 划分（30% 留出）：train 749 个函数，heldout 321 个（完整版新增 156、sanitized 121、HumanEval 44）。

## 近似重复检查的结果

方法按修订 2：对每个来源函数的入口函数，去掉 docstring，把参数名和所有非内置的名字都换成同一个占位符，`ast.unparse` 后用 `tokenize` 切成 token，取 3-gram 集合，计算每个留出集函数和全部训练集函数的最大 Jaccard。检查对象是加载后的全部 1070 个来源函数（在变异筛查之前）。

### 移动数量和阈值的关系

| 阈值（最大 Jaccard >） | 留出集函数数 | 占 321 的比例 |
| --- | --- | --- |
| 0.6 | 125 | 38.9% |
| 0.7 | 89 | 27.7% |
| **0.8（修订 2 规定）** | **63** | **19.6%** |
| 0.9 | 20 | 6.2% |
| 0.95 | 17 | 5.3% |
| 规范化后完全相同（1.0） | 17 | 5.3% |

### 重复来自哪里（阈值 0.8 的 63 个）

| 留出集函数来源 → 最相似的训练集函数来源 | 数量 |
| --- | --- |
| MBPP 完整版新增 → MBPP 完整版新增 | 24 |
| MBPP sanitized → MBPP 完整版新增 | 18 |
| MBPP 完整版新增 → MBPP sanitized | 13 |
| MBPP sanitized → MBPP sanitized | 7 |
| HumanEval → MBPP 完整版新增 | 1 |

- 按留出集函数的来源算：MBPP 完整版新增 37/156（23.7%），MBPP sanitized 25/121（20.7%），HumanEval 1/44（2.3%）。
- 对照：只用之前的 540 个函数（sanitized 加 HumanEval），同样的 seed 和方法，是 10/162（6.2%），在门槛以内。所以超标主要是加入完整版 MBPP 带来的。MBPP 本身就有不少重复题（同名的 `max_occurrences`、`remove_empty`、`word_len`、`snake_to_camel`、`change_date_format` 等各出现两次），完整版把它们带进来了。
- 在全部 1070 个函数之间两两比较：Jaccard 超过 0.8 的函数对有 173 对，按"相似就连边"连起来，有 216 个函数落在 91 个多成员的簇里，最大的簇有 9 个函数。

### 例子（0.8–0.9 之间随机抽的）

```python
# mbpp/202（heldout）                         # mbpp/437（train），Jaccard 0.889
def remove_even(str1):                        def remove_odd(str1):
    str2 = ''                                     str2 = ''
    for i in range(1, len(str1) + 1):             for i in range(1, len(str1) + 1):
        if i % 2 != 0:                                if i % 2 == 0:
            str2 = str2 + str1[i - 1]                     str2 = str2 + str1[i - 1]
    return str2                                   return str2

# mbpp/831（heldout）                         # mbpp/841（train），Jaccard 0.885
def count_Pairs(arr, n):                      def get_inv_count(arr, n):
    cnt = 0                                       inv_count = 0
    for i in range(n):                            for i in range(n):
        for j in range(i + 1, n):                     for j in range(i + 1, n):
            if arr[i] == arr[j]:                          if arr[i] > arr[j]:
                cnt += 1                                      inv_count += 1
    return cnt                                    return inv_count

# mbpp/429（heldout）                         # mbpp/197（train），Jaccard 0.818
def and_tuples(test_tup1, test_tup2):         def find_exponentio(test_tup1, test_tup2):
    res = tuple((ele1 & ele2 for ...))            res = tuple((ele1 ** ele2 for ...))
    return res                                    return res
```

完整清单（63 个）见本文末尾的附录。

## 需要审计方决定的事

以下方案我都没有采用：

1. **按簇划分**：先在全部来源函数上把 Jaccard > 0.8 的函数连成簇（上面统计的 91 个簇、216 个函数），以簇为单位做 train/heldout 划分，然后照常跑一遍检查（按构造，移动数应该接近 0）。这是处理数据泄漏的常见做法，留出集比例不变。但这改变了修订 2 规定的"先划分、再把重复的移到训练集"的流程。
2. **照常移动 63 个，忽略 10% 门槛**：留出集剩 258 个函数，训练集 812 个。
3. **提高阈值**：比如只移动规范化后完全相同的 17 个（5.3%），或者阈值取 0.9（20 个，6.2%）。0.8–0.9 之间的对子，从上面的例子看多数是真正的"同一模板换一个运算符"，提高阈值会把这类泄漏留在留出集里。
4. **不用完整版 MBPP 新增的部分**：回到 540 个函数，近似重复 6.2%，在门槛内。但这样又回到 CHECKPOINT-1B 的题量，修订 3 的新门槛（校准后留出集至少 200 题、120 个不同 bug 来源函数）能不能满足要等校准结果。

## 隐藏用例加强：已实现，试跑结果

还没接进构建，但代码和测试已经完成，等近似重复的问题定下来再接。规则按修订 2 第 4 条：

- 以每个已有用例的参数为种子，每个候选只扰动 1 个参数（30% 的概率扰动 2 个）。整数：加减 1–3、取负、置 0、变成负数；浮点：加减小量、取负、置 0；布尔：取反；字符串、列表、元组：清空、截断、删一个元素、插入一个元素（字符串插入它本身已有的字符，列表插入一个已有元素的扰动版）、交换两个元素、扰动其中一个元素（保持嵌套结构）；字典：删一个键、扰动一个值、清空；集合：删除或插入元素。容器最多比种子长 5。
- 每个函数用固定 seed（`augment-{source_id}`）生成最多 25 个不重复的候选，并排除和已有用例（包括可见测试的候选）完全相同的参数。
- 每个候选在沙箱里用原始代码跑两次（两个独立进程，每条限时 1 秒）。两次都正常返回、结果相同、能用 `repr` 加 `ast.literal_eval` 还原且类型逐层一致，才保留。抛异常、超时、不确定、不能还原的都丢弃并计数。浮点结果用 `math.isclose`；容器里含浮点、复数结果、期望值超过 500 字符的也丢弃。每个函数最多保留 10 条。
- 集合按排序后的元素输出，宿主机的哈希随机化不会让构建结果不确定。
- 生成的用例记为 `kind="generated"`，只进隐藏用例；可见测试只能从 `kind="test"` 里选。筛查规则相应调整：变体必须至少挂 1 条 `test` 用例（当可见测试），并且总共至少挂 2 条。
- 还没做的：bug 函数隐藏用例不足 5 条的题丢弃并计数、整题重新验证、加强前后的分布统计、补出的用例让 bug 版失败的比例。这些都在接进构建时一起做。

试跑（随机 60 个来源函数，8 线程，23 秒）：

| 项 | 数量 |
| --- | --- |
| 候选输入（当时上限是 40，之后为了速度改成 25） | 2153 |
| 保留（每函数上限 10） | 595（59 个函数补满 10 条，1 个函数补了 5 条） |
| 原始代码抛异常 | 182 |
| 超时 | 11 |
| 合格但超出 10 条上限 | 1365 |

例子（`mbpp/88 freq_count`，原测试是 13 个元素的列表）：`[]` → `{}`；`[10, 10, 10, 0, 20, ...]` → `{10: 3, 0: 1, 20: 4, ...}`。

**观察到的风险**：扰动后的输入可能超出原题的预期用法，比如 `max_sub_array_sum([-1, -2, 3], -7)`，第二个参数本来应该是列表长度。期望值来自原始代码的实际行为，所以把 bug 改回原样一定能通过；但如果模型重写了整个函数，写法不同但对正常输入都正确，在这类输入上可能拿不到奖励。

## 附录：阈值 0.8 下的 63 个留出集函数

| 留出集函数 | 来源 | 最大 Jaccard | 最相似的训练集函数 |
| --- | --- | --- | --- |
| `mbpp/10` small_nnum | mbpp_full | 1.0 | `mbpp/496` heap_queue_smallest |
| `mbpp/17` square_perimeter | mbpp_sanitized | 1.0 | `mbpp/716` rombus_perimeter |
| `mbpp/45` get_gcd | mbpp_full | 1.0 | `mbpp/520` get_lcm |
| `mbpp/116` tuple_to_int | mbpp_sanitized | 1.0 | `mbpp/439` multiple_to_single |
| `mbpp/178` string_literals | mbpp_full | 1.0 | `mbpp/186` check_literals |
| `mbpp/338` count_Substring_With_Equal_Ends | mbpp_full | 1.0 | `mbpp/30` count_Substring_With_Equal_Ends |
| `mbpp/413` extract_nth_element | mbpp_sanitized | 1.0 | `mbpp/49` specified_element |
| `mbpp/458` rectangle_area | mbpp_sanitized | 1.0 | `mbpp/52` parallelogram_area |
| `mbpp/542` fill_spaces | mbpp_full | 1.0 | `mbpp/732` replace_specialchar |
| `mbpp/649` sum_Range_list | mbpp_full | 1.0 | `mbpp/170` sum_range_list |
| `mbpp/658` max_occurrences | mbpp_full | 1.0 | `mbpp/362` max_occurrences |
| `mbpp/693` remove_multiple_spaces | mbpp_full | 1.0 | `mbpp/892` remove_spaces |
| `mbpp/703` is_key_present | mbpp_full | 1.0 | `mbpp/405` check_tuplex |
| `mbpp/800` remove_all_spaces | mbpp_sanitized | 1.0 | `mbpp/451` remove_whitespaces |
| `mbpp/825` access_elements | mbpp_full | 1.0 | `mbpp/49` specified_element |
| `mbpp/842` get_odd_occurence | mbpp_full | 1.0 | `mbpp/29` get_Odd_Occurrence |
| `mbpp/966` remove_empty | mbpp_full | 1.0 | `mbpp/361` remove_empty |
| `mbpp/752` jacobsthal_num | mbpp_sanitized | 0.912 | `mbpp/529` jacobsthal_lucas |
| `mbpp/383` even_bit_toggle_number | mbpp_full | 0.911 | `mbpp/155` even_bit_toggle_number |
| `mbpp/855` check_Even_Parity | mbpp_full | 0.902 | `mbpp/768` check_Odd_Parity |
| `mbpp/373` volume_cuboid | mbpp_full | 0.9 | `mbpp/52` parallelogram_area |
| `mbpp/942` check_element | mbpp_full | 0.895 | `mbpp/808` check_K |
| `mbpp/442` positive_count | mbpp_sanitized | 0.893 | `mbpp/530` negative_count |
| `mbpp/168` frequency | mbpp_sanitized | 0.889 | `mbpp/446` count_Occurrence |
| `mbpp/202` remove_even | mbpp_full | 0.889 | `mbpp/437` remove_odd |
| `mbpp/564` count_Pairs | mbpp_sanitized | 0.885 | `mbpp/841` get_inv_count |
| `mbpp/831` count_Pairs | mbpp_full | 0.885 | `mbpp/841` get_inv_count |
| `mbpp/964` word_len | mbpp_full | 0.872 | `mbpp/79` word_len |
| `mbpp/327` check_isosceles | mbpp_full | 0.871 | `mbpp/51` check_equilateral |
| `mbpp/682` mul_list | mbpp_full | 0.867 | `mbpp/865` ntimes_list |
| `mbpp/440` find_adverb_position | mbpp_sanitized | 0.86 | `mbpp/584` find_adverbs |
| `mbpp/345` diff_consecutivenums | mbpp_full | 0.854 | `mbpp/503` add_consecutive_nums |
| `mbpp/389` find_lucas | mbpp_sanitized | 0.854 | `mbpp/960` get_noOfways |
| `mbpp/690` mul_consecutive_nums | mbpp_full | 0.854 | `mbpp/503` add_consecutive_nums |
| `mbpp/517` largest_pos | mbpp_full | 0.846 | `mbpp/443` largest_neg |
| `mbpp/726` multiply_elements | mbpp_sanitized | 0.842 | `mbpp/206` concatenate_elements |
| `mbpp/290` max_length | mbpp_sanitized | 0.838 | `mbpp/528` min_length |
| `mbpp/412` remove_odd | mbpp_sanitized | 0.838 | `mbpp/824` remove_even |
| `mbpp/888` substract_elements | mbpp_full | 0.838 | `mbpp/401` add_nested_tuples |
| `mbpp/452` loss_amount | mbpp_sanitized | 0.833 | `mbpp/954` profit_amount |
| `mbpp/875` min_difference | mbpp_full | 0.833 | `mbpp/63` max_difference |
| `mbpp/913` end_num | mbpp_full | 0.833 | `mbpp/900` match_num |
| `mbpp/258` count_odd | mbpp_full | 0.829 | `mbpp/697` count_even |
| `mbpp/807` first_odd | mbpp_sanitized | 0.829 | `mbpp/221` first_even |
| `mbpp/928` change_date_format | mbpp_full | 0.826 | `mbpp/427` change_date_format |
| `mbpp/932` remove_duplic_list | mbpp_full | 0.825 | `mbpp/436` neg_nos |
| `mbpp/285` text_match_two_three | mbpp_sanitized | 0.824 | `mbpp/16` text_lowercase_underscore |
| `mbpp/917` text_uppercase_lowercase | mbpp_full | 0.824 | `mbpp/43` text_match |
| `humaneval/63` fibfib | humaneval | 0.822 | `mbpp/960` get_noOfways |
| `mbpp/294` max_val | mbpp_sanitized | 0.818 | `mbpp/410` min_val |
| `mbpp/399` bitwise_xor | mbpp_sanitized | 0.818 | `mbpp/197` find_exponentio |
| `mbpp/429` and_tuples | mbpp_sanitized | 0.818 | `mbpp/197` find_exponentio |
| `mbpp/731` lateralsurface_cone | mbpp_sanitized | 0.816 | `mbpp/497` surfacearea_cone |
| `mbpp/402` ncr_modp | mbpp_full | 0.813 | `mbpp/952` nCr_mod_p |
| `mbpp/41` filter_evennumbers | mbpp_full | 0.812 | `mbpp/426` filter_oddnumbers |
| `mbpp/282` sub_list | mbpp_sanitized | 0.812 | `mbpp/358` moddiv_list |
| `mbpp/919` multiply_list | mbpp_full | 0.812 | `mbpp/925` mutiple_tuple |
| `mbpp/411` snake_to_camel | mbpp_sanitized | 0.811 | `mbpp/102` snake_to_camel |
| `mbpp/573` unique_product | mbpp_sanitized | 0.811 | `mbpp/925` mutiple_tuple |
| `mbpp/654` rectangle_perimeter | mbpp_full | 0.808 | `mbpp/369` lateralsurface_cuboid |
| `mbpp/645` find_k_product | mbpp_full | 0.806 | `mbpp/611` max_of_nth |
| `mbpp/549` odd_Num_Sum | mbpp_full | 0.804 | `mbpp/655` fifth_Power_Sum |
| `mbpp/770` odd_num_sum | mbpp_sanitized | 0.804 | `mbpp/655` fifth_Power_Sum |
