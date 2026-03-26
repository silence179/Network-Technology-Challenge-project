// tb_sequence.v
// 仿真激励：data = 'b11011_0111_0111（13位，高位先输入）
// 序列展开：1 1 0 1 1 0 1 1 1 0 1 1 1
// 同时仿真Moore型和Mealy型自动机

`timescale 1ns/1ps

module tb_sequence;

    reg        clk;
    reg        rst;
    reg        din;
    wire       moore_out;
    wire       mealy_out;

    // 例化Moore型自动机
    sequence_moore u_moore (
        .clk  (clk),
        .rst  (rst),
        .din  (din),
        .dout (moore_out)
    );

    // 例化Mealy型自动机
    sequence_mealy u_mealy (
        .clk  (clk),
        .rst  (rst),
        .din  (din),
        .dout (mealy_out)
    );

    // 时钟：周期10ns
    initial clk = 0;
    always #5 clk = ~clk;

    // 仿真数据：data = 'b11011_0111_0111
    reg [12:0] data = 13'b1101101110111;
    integer i;

    initial begin
        $dumpfile("tb_sequence.vcd");
        $dumpvars(0, tb_sequence);

        // 复位
        rst = 1;
        din = 0;
        @(negedge clk); #1;
        @(negedge clk); #1;
        rst = 0;

        // 从高位到低位逐位输入
        for (i = 12; i >= 0; i = i - 1) begin
            @(negedge clk); #1;
            din = data[i];
        end

        // 额外空闲周期
        @(negedge clk); #1; din = 0;
        @(negedge clk); #1; din = 0;

        #20;
        $finish;
    end

    // 监视输出
    initial begin
        $display("Time  clk rst din | Moore_state Moore_out | Mealy_state Mealy_out");
        $monitor("%4t   %b   %b   %b  |     %b          %b     |     %b          %b",
                 $time, clk, rst, din,
                 u_moore.state, moore_out,
                 u_mealy.state, mealy_out);
    end

endmodule
