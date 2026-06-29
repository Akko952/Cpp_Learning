#include <iostream>

using namespace std;

class Line {
  public:
    Line(double len); // 带参构造函数声明
    void setLength(double len);
    double getLength() const;

  private:
    double length;
};

// 构造函数定义
Line::Line(double len) {
    cout << "Object is being created, length = " << len << endl;
    length = len;
}

void Line::setLength(double len) { length = len; }

double Line::getLength() const { return length; }

// 程序的主函数
int main() {
    Line line(10.0); // 创建对象时直接传入初始长度

    // 获取初始长度
    cout << "Length of line : " << line.getLength() << endl;
    // 重新设置长度
    line.setLength(6.0);
    cout << "Length of line : " << line.getLength() << endl;

    return 0;
}